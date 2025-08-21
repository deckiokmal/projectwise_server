# app/services/llm//llm_chains.py
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable
from typing import Any, Dict, List, Union, Type, Tuple, Optional, Protocol

from pydantic import BaseModel, ValidationError

from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    BadRequestError,
    AuthenticationError,
    InternalServerError,
)

from mcp_server.settings import Settings
from mcp_server.utils.logger import get_logger

logger = get_logger(__name__)
settings = Settings()  # type: ignore


class ToolExecutor(Protocol):
    def __call__(
        self, name: str, args: Dict[str, Any]
    ) -> Awaitable[Dict[str, Any]] | Dict[str, Any]: ...


class LLMChains:
    """
    Utilitas LLM serbaguna (full async) yang menyediakan:
    - Chat Completions (JSON Schema)
    - Responses API (Pydantic schema)
    - Function Call (tools) dengan fallback
    - Parsing structured output (Pydantic) secara defensif

    Seluruh method mengembalikan dict konsisten minimal:
        { "status": "success"|"error", "message": str, ... }
    """

    def __init__(
        self,
        model: str = settings.llm_model,
        temperature: float = settings.llm_temperature,
        max_tokens: int = 2048,
        api_key: Optional[str] = settings.llm_api_key,
        timeout_sec: float = 60.0,
    ):
        """
        Inisialisasi klien OpenAI async dan parameter default.

        Args:
            model: Nama model default.
            temperature: Nilai temperature.
            max_tokens: Batas token output (bila didukung).
            api_key: API key OpenAI.
            timeout_sec: Batas waktu (detik) tiap request.
        """
        self.model = model
        self.temperature = float(temperature or 0.0)
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec
        self.client = AsyncOpenAI(api_key=api_key)

    # ======================================================================
    # -------------------------- UTIL INTERNAL ------------------------------
    # ======================================================================

    def _ret(self, status: str, message: str, **extra: Any) -> Dict[str, Any]:
        """
        Builder untuk respons konsisten.

        Returns:
            Dict minimal memiliki status & message.
        """
        base = {"status": status, "message": message}
        base.update(extra or {})
        return base

    def _json_schema_rf(
        self, name: Optional[str], schema: Dict[str, Any], strict: bool = True
    ) -> Dict[str, Any]:
        """
        Bangun 'response_format' untuk Chat Completions berbasis JSON Schema.
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name or f"schema_{uuid.uuid4().hex[:8]}",
                "schema": schema,
                "strict": strict,
            },
        }

    def _normalize_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Normalisasi schema tools agar kompatibel dengan OpenAI:
        - pastikan parameters.additionalProperties = False (bila tidak ada).
        """
        safe: List[Dict[str, Any]] = []
        for t in tools or []:
            try:
                if t.get("type") == "function":
                    params = t.get("function", {}).get("parameters", {})
                    if (
                        isinstance(params, dict)
                        and "additionalProperties" not in params
                    ):
                        params = {**params, "additionalProperties": False}
                        t = {
                            **t,
                            "function": {
                                **t.get("function", {}),
                                "parameters": params,
                            },
                        }
                safe.append(t)
            except Exception:
                logger.warning("Gagal menormalkan tool: %s", t, exc_info=True)
                safe.append(t)
        return safe

    def _extract_calls_from_chat(self, resp) -> List[Dict[str, Any]]:
        """
        Ekstraksi function call dari Chat Completions.
        """
        calls: List[Dict[str, Any]] = []
        try:
            choice = (resp.choices or [None])[0]
            msg = getattr(choice, "message", None)
            for tc in getattr(msg, "tool_calls", []) or []:
                fn = getattr(tc, "function", None)
                args_str = getattr(fn, "arguments", "") or "{}"
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {"_raw": args_str}
                calls.append(
                    {
                        "name": getattr(fn, "name", ""),
                        "arguments": args,
                        "id": getattr(tc, "id", None),
                        "raw": tc,
                    }
                )
        except Exception:
            logger.exception("Gagal mengekstrak tool_calls (chat).")
        return calls

    def _extract_calls_from_responses(self, resp) -> List[Dict[str, Any]]:
        """
        Ekstraksi function call dari Responses API.
        """
        calls: List[Dict[str, Any]] = []
        try:
            output = getattr(resp, "output", None)
            if isinstance(output, list):
                for it in output:
                    if getattr(it, "type", None) == "function_call":
                        args = getattr(it, "arguments", "") or "{}"
                        try:
                            args_json = json.loads(args)
                        except Exception:
                            args_json = {"_raw": args}
                        calls.append(
                            {
                                "name": getattr(it, "name", ""),
                                "arguments": args_json,
                                "call_id": getattr(it, "call_id", None),
                                "raw": it,
                            }
                        )
        except Exception:
            logger.exception("Gagal mengekstrak function_call (responses).")
        return calls

    def _friendly_error(self, e: Exception) -> str:
        """
        Konversi exception menjadi pesan ramah pengguna.
        """
        if isinstance(e, (APIConnectionError, APITimeoutError)):
            return "Koneksi ke LLM bermasalah atau timeout. Coba ulangi."
        if isinstance(e, RateLimitError):
            return "Kutipan penggunaan model terlampaui (rate limit). Coba beberapa saat lagi."
        if isinstance(e, AuthenticationError):
            return "Autentikasi LLM gagal. Periksa API key."
        if isinstance(e, BadRequestError):
            return "Permintaan ke LLM ditolak. Periksa format input/parameter."
        if isinstance(e, InternalServerError):
            return "Layanan LLM mengalami gangguan. Coba ulangi."
        return f"Kesalahan tidak terduga: {e}"

    async def _maybe_await(self, value):
        """Util: menerima sync/async return dan mengembalikan hasilnya secara async."""
        if asyncio.iscoroutine(value):
            return await value
        return value

    def _as_chat_tool_calls_dicts(
        self, calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalisasi daftar 'calls' (hasil _extract_calls_from_chat) menjadi struktur
        tool_calls yang valid untuk Chat Completions (JSON-serializable).
        """
        out = []
        for c in calls:
            name = c.get("name", "")
            args = c.get("arguments", {}) or {}
            call_id = c.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            # Pastikan arguments string JSON
            args_str = (
                json.dumps(args, ensure_ascii=False)
                if isinstance(args, (dict, list))
                else str(args or "{}")
            )
            out.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_str},
                }
            )
        return out

    def _to_responses_input(
        self, chat_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Konversi messages gaya Chat ke 'input' gaya Responses API yang JSON-serializable.
        - content:str → [{"type":"text","text":...}]
        - assistant.tool_calls → [{"type":"function_call", "name","arguments","call_id"}]
        - tool message → {"role":"tool","tool_call_id":..., "name":..., "content":[{"type":"output_text","text":...}]}
        """
        resp_input: List[Dict[str, Any]] = []
        for m in chat_messages:
            role = m.get("role")
            # 1) Pesan TOOL (dari chat) → item tool untuk responses
            if role == "tool":
                tool_call_id = m.get("tool_call_id")
                name = m.get("name")
                content = m.get("content", "")
                resp_input.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "content": [
                            {
                                "type": "output_text",
                                "text": content
                                if isinstance(content, str)
                                else json.dumps(content, ensure_ascii=False),
                            }
                        ],
                    }
                )
                continue

            # 2) Assistant yang berisi tool_calls → function_call items
            tool_calls = m.get("tool_calls") or []
            if role == "assistant" and tool_calls:
                items = []
                for tc in tool_calls:
                    # tc mungkin object dari SDK → ambil sebagai dict aman
                    fn = getattr(tc, "function", None)
                    tc_id = (
                        getattr(tc, "id", None) or tc.get("id")
                        if isinstance(tc, dict)
                        else None
                    )
                    if fn is not None:
                        fn_name = getattr(fn, "name", None)
                        fn_args = getattr(fn, "arguments", None)
                    else:
                        # dict path
                        fn_name = (
                            tc.get("function", {}).get("name")
                            if isinstance(tc, dict)
                            else None
                        )
                        fn_args = (
                            tc.get("function", {}).get("arguments")
                            if isinstance(tc, dict)
                            else None
                        )

                    # pastikan string JSON
                    if isinstance(fn_args, (dict, list)):
                        fn_args = json.dumps(fn_args, ensure_ascii=False)
                    elif not isinstance(fn_args, str):
                        fn_args = "{}"

                    items.append(
                        {
                            "type": "function_call",
                            "name": fn_name or "",
                            "arguments": fn_args or "{}",
                            "call_id": tc_id or f"call_{uuid.uuid4().hex[:8]}",
                        }
                    )
                resp_input.append({"role": "assistant", "content": items})
                continue

            # 3) Pesan biasa (system/user/assistant tanpa tool_calls)
            content = m.get("content")
            if isinstance(content, str):
                resp_input.append(
                    {"role": role, "content": [{"type": "text", "text": content}]}
                )
            elif isinstance(content, list):
                # Sudah berbentuk list content → gunakan langsung jika JSON-serializable
                resp_input.append({"role": role, "content": content})
            else:
                # Fallback ke teks
                resp_input.append(
                    {"role": role, "content": [{"type": "text", "text": str(content)}]}
                )
        return resp_input

    def _ensure_responses_input(
        self, maybe_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Jika sudah responses-style (role+content:list), kembalikan apa adanya.
        Jika chat-style (content:str / tool_calls), ubah ke responses-style.
        """
        if not maybe_messages:
            return []
        first = maybe_messages[0]
        if isinstance(first.get("content"), list):
            return maybe_messages
        return self._to_responses_input(maybe_messages)

    async def _chat_request_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
    ):
        """Panggil Chat Completions dengan tools (1-hop)."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return await asyncio.wait_for(
            self.client.chat.completions.create(**kwargs), timeout=self.timeout_sec
        )

    # ======================================================================
    # ------------------------ CHAT COMPLETIONS -----------------------------
    # ======================================================================

    async def chat_completions_text(
        self,
        messages: List[Dict[str, Any]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Panggil Chat Completions untuk menghasilkan text (opsional: JSON Schema).

        Args:
            messages: daftar pesan (role: system|user|assistant).
            json_schema: skema JSON untuk structured output (opsional).
            max_tokens: override batas token.

        Returns:
            Dict konsisten {status, message, data?, raw?}
        """
        logger.info("ChatCompletions: generate text (schema=%s)", bool(json_schema))
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if max_tokens or self.max_tokens:
                kwargs["max_tokens"] = max_tokens or self.max_tokens
            if json_schema:
                kwargs["response_format"] = self._json_schema_rf(
                    name=(json_schema.get("$id") or "chat_schema"),
                    schema=json_schema,
                    strict=True,
                )

            resp = await asyncio.wait_for(
                self.client.chat.completions.create(**kwargs),
                timeout=self.timeout_sec,
            )

            choice = (resp.choices or [None])[0]
            content = getattr(getattr(choice, "message", None), "content", None)

            data: Any = content
            # Bila pakai JSON Schema, content kemungkinan adalah string JSON -> coba parse
            if json_schema and isinstance(content, str):
                try:
                    data = json.loads(content)
                except Exception:
                    logger.warning(
                        "Structured content bukan JSON valid, mengembalikan string mentah."
                    )

            return self._ret(
                "success",
                "Berhasil menghasilkan teks (chat).",
                data=data,
                raw=resp,
                meta={"endpoint": "chat.completions"},
            )
        except Exception as e:
            logger.error("ChatCompletions gagal: %s", e, exc_info=True)
            return self._ret("error", self._friendly_error(e))

    # ======================================================================
    # --------------------------- RESPONSES API -----------------------------
    # ======================================================================

    async def responses_text(
        self,
        input: Union[str, List[Dict[str, Any]]],
        *,
        pydantic_model: Optional[Type[BaseModel]] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Panggil Responses API untuk menghasilkan text atau structured (Pydantic).

        Args:
            input: teks atau daftar messages ala Responses (role+content).
            pydantic_model: model Pydantic target untuk structured output.
            max_output_tokens: override batas token output.

        Returns:
            Dict konsisten {status, message, data?, raw?}
        """
        logger.info("Responses: generate (pydantic=%s)", bool(pydantic_model))
        try:
            # 1) Coba gunakan responses.parse jika tersedia (lebih aman untuk Pydantic)
            if pydantic_model is not None and hasattr(self.client.responses, "parse"):
                logger.debug("Responses.parse → Pydantic")
                parsed = await asyncio.wait_for(
                    self.client.responses.parse(
                        model=self.model,
                        input=input,
                        temperature=self.temperature,
                        response_format=pydantic_model,  # type: ignore[arg-type]
                        max_output_tokens=max_output_tokens or self.max_tokens,
                    ),
                    timeout=self.timeout_sec,
                )
                # parsed adalah instance pydantic (atau mirip), dump jadi dict
                data = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
                return self._ret(
                    "success",
                    "Berhasil menghasilkan structured output (responses.parse).",
                    data=data,
                    raw=None,
                    meta={"endpoint": "responses.parse"},
                )

            # 2) Tanpa parse(): gunakan responses.create
            logger.debug("Responses.create")
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": input,
                "temperature": self.temperature,
            }
            if max_output_tokens or self.max_tokens:
                kwargs["max_output_tokens"] = max_output_tokens or self.max_tokens

            # Bila minta Pydantic: injeksikan JSON Schema dari Pydantic agar model diarahkan
            if pydantic_model is not None:
                try:
                    schema = pydantic_model.model_json_schema()  # type: ignore[attr-defined]
                except Exception:
                    schema = {}  # fallback—tetap jalan, nanti validasi manual di bawah
                kwargs["response_format"] = self._json_schema_rf(
                    name=pydantic_model.__name__,
                    schema=schema or {"type": "object"},
                    strict=True,
                )

            resp = await asyncio.wait_for(
                self.client.responses.create(**kwargs),
                timeout=self.timeout_sec,
            )

            # Ambil output_text (bisa kosong bila ada tool call)
            text = (getattr(resp, "output_text", "") or "").strip()

            # Jika ada pydantic_model → coba validasi manual dari text
            if pydantic_model is not None and text:
                try:
                    data_obj = pydantic_model.model_validate_json(text)  # type: ignore[attr-defined]
                    data = data_obj.model_dump()  # type: ignore[attr-defined]
                    return self._ret(
                        "success",
                        "Berhasil menghasilkan structured output (responses + validate).",
                        data=data,
                        raw=resp,
                        meta={"endpoint": "responses.create"},
                    )
                except ValidationError as ve:
                    logger.error("Validasi Pydantic gagal: %s", ve)
                    return self._ret(
                        "error",
                        f"Parsing structured output gagal: {ve.errors()}",
                        raw=resp,
                        meta={"endpoint": "responses.create"},
                    )

            # Tanpa pydantic → kembalikan text biasa
            return self._ret(
                "success",
                "Berhasil menghasilkan teks (responses).",
                data=text,
                raw=resp,
                meta={"endpoint": "responses.create"},
            )

        except Exception as e:
            logger.error("Responses gagal: %s", e, exc_info=True)
            return self._ret("error", self._friendly_error(e))

    async def responses_parse(
        self,
        input: Union[str, List[Dict[str, Any]]],
        *,
        pydantic_model: Type[BaseModel],
    ) -> Dict[str, Any]:
        """
        Paksa structured output via Responses API + Pydantic.

        Prefer responses.parse bila tersedia, else fallback ke responses.create
        lalu validasi manual.
        """
        logger.info("Responses.parse (prefer) → %s", pydantic_model.__name__)
        # Reuse responses_text dengan pydantic_model
        return await self.responses_text(input=input, pydantic_model=pydantic_model)

    # ======================================================================
    # --------------------------- FUNCTION CALL -----------------------------
    # ======================================================================

    async def function_call(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: List[Dict[str, Any]],
        prefer: str = "chat",
        tool_choice: Union[str, Dict[str, Any]] = "auto",
    ) -> Dict[str, Any]:
        """
        Jalankan function-calling dengan fallback:
        - prefer "chat" → coba Chat (tools), jika gagal/kosong → fallback Responses.
        - prefer "chat" → sebaliknya.

        Args:
            messages: daftar messages (system|user|assistant).
            tools: daftar tools OpenAI (type=function, parameters=jsonschema).
            prefer: "responses"|"chat"
            tool_choice: "auto"|"required"|{"type":"function","function":{"name":...}}

        Returns:
            Dict konsisten {status, message, calls?, raw?, meta?}
        """
        tools_norm = self._normalize_tools(tools)

        async def _via_responses() -> Tuple[List[Dict[str, Any]], Any]:
            try:
                logger.info("FunctionCall via Responses")
                input_payload = self._ensure_responses_input(
                    messages
                )  # ← normalisasi DI SINI
                resp = await asyncio.wait_for(
                    self.client.responses.create(
                        model=self.model,
                        input=input_payload,  # type: ignore
                        tools=tools_norm,  # type: ignore[arg-type]
                        tool_choice=tool_choice,  # type: ignore
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                    timeout=self.timeout_sec,
                )
                calls = self._extract_calls_from_responses(resp)
                return calls, resp
            except Exception as e:
                logger.error("FunctionCall Responses gagal: %s", e, exc_info=True)
                return [], e

        async def _via_chat() -> Tuple[List[Dict[str, Any]], Any]:
            try:
                logger.info("FunctionCall via Chat Completions")
                resp = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,  # type: ignore
                        tools=tools_norm,  # type: ignore[arg-type]
                        tool_choice=tool_choice,  # type: ignore
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    ),
                    timeout=self.timeout_sec,
                )
                calls = self._extract_calls_from_chat(resp)
                return calls, resp
            except Exception as e:
                logger.error("FunctionCall Chat gagal: %s", e, exc_info=True)
                return [], e

        order = (
            (_via_responses, _via_chat)
            if prefer == "responses"
            else (_via_chat, _via_responses)
        )

        # Coba jalur preferensi dulu
        calls, raw = await order[0]()
        if calls:
            return self._ret(
                "success",
                "Berhasil mengekstrak function call.",
                calls=calls,
                raw=raw,
                meta={
                    "endpoint": "responses"
                    if order[0] is _via_responses
                    else "chat.completions"
                },
            )

        # Fallback ke jalur kedua
        calls_fb, raw_fb = await order[1]()
        if calls_fb:
            return self._ret(
                "success",
                "Berhasil mengekstrak function call (fallback).",
                calls=calls_fb,
                raw=raw_fb,
                meta={
                    "endpoint": "chat.completions"
                    if order[0] is _via_responses
                    else "responses"
                },
            )

        # Keduanya gagal / kosong
        if isinstance(raw, Exception):
            msg = self._friendly_error(raw)
        elif isinstance(raw_fb, Exception):
            msg = self._friendly_error(raw_fb)
        else:
            msg = "Tidak ditemukan function call dari kedua endpoint."
        return self._ret("error", msg, meta={"endpoint": "fallback-failed"})

    async def run_function_call_roundtrip(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tool_executor: ToolExecutor,
        prefer: str = "chat",
        max_hops: int = 4,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> dict[str, Any]:
        """
        Jalankan *function-calling roundtrip* end-to-end hingga final text:
        1) Minta model → tool_calls
        2) Eksekusi fungsi di backend (via tool_executor)
        3) Suntik hasil ke percakapan
        4) Ulangi sampai tak ada tool_calls → ambil jawaban final

        Args:
            messages: pesan awal gaya Chat Completions
            tools: daftar schema tools (JSON Schema)
            tool_executor: callable untuk mengeksekusi fungsi (boleh sync/async)
            prefer: 'chat' (produksi) atau 'responses' (opsional)
            max_hops: maksimum iterasi tool-call

        Returns:
            dict {status, message, data?, meta?}
        """
        tools_norm = self._normalize_tools(tools)

        if prefer == "responses":
            # Jalur opsional: normalisasi ke responses 'input'
            input_msgs = self._ensure_responses_input(messages)
            for hop in range(1, max_hops + 1):
                # 1) minta function_call
                try:
                    resp = await asyncio.wait_for(
                        self.client.responses.create(
                            model=self.model,
                            input=input_msgs,  # type: ignore
                            tools=tools_norm,  # type: ignore[arg-type]
                            tool_choice=tool_choice,  # type: ignore
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        ),
                        timeout=self.timeout_sec,
                    )
                except Exception as e:
                    return self._ret("error", self._friendly_error(e))

                calls = self._extract_calls_from_responses(resp)
                if not calls:
                    # Final text
                    text = (getattr(resp, "output_text", "") or "").strip()
                    return self._ret(
                        "success",
                        "Final response (responses).",
                        data=text,
                        raw=resp,
                        meta={"hops": hop},
                    )

                # 2) eksekusi tiap call (sekuensial; bisa Anda ubah jadi paralel)
                fc_items = []
                tool_items = []
                for c in calls:
                    name = c["name"]
                    args = c.get("arguments") or {}
                    call_id = c.get("call_id") or f"call_{uuid.uuid4().hex[:8]}"

                    result = await self._maybe_await(tool_executor(name, args))
                    # assistant: function_call item
                    fc_items.append(
                        {
                            "type": "function_call",
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False)
                            if isinstance(args, (dict, list))
                            else str(args or "{}"),
                            "call_id": call_id,
                        }
                    )

                    # tool result
                    if not isinstance(result, dict):
                        result = {"status": "success", "message": "ok", "data": result}
                    result.setdefault("status", "success")
                    result.setdefault("message", "ok")

                    tool_items.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        result, ensure_ascii=False, default=str
                                    ),
                                }
                            ],
                        }
                    )

                # 3) jahit hasil ke input (responses-style)
                input_msgs = [
                    *input_msgs,
                    {"role": "assistant", "content": fc_items},
                    *tool_items,
                ]

                logger.info(
                    "Roundtrip hop=%s prefer=%s (calls=%d)", hop, prefer, len(calls)
                )

            return self._ret(
                "error",
                f"Max hops tercapai ({max_hops}) tanpa final response (responses).",
            )

        # === prefer == "chat": jalur produksi ===
        chat_msgs = list(messages)
        for hop in range(1, max_hops + 1):
            try:
                resp = await self._chat_request_with_tools(
                    chat_msgs, tools_norm, tool_choice=tool_choice
                )
            except Exception as e:
                return self._ret("error", self._friendly_error(e))

            calls = self._extract_calls_from_chat(resp)
            if not calls:
                # Tidak ada tool_calls → ambil final text
                choice = (resp.choices or [None])[0]
                final_text = (
                    getattr(getattr(choice, "message", None), "content", "") or ""
                )
                return self._ret(
                    "success",
                    "Final response (chat).",
                    data=final_text,
                    raw=resp,
                    meta={"hops": hop},
                )

            # Ada satu/lebih tool_calls → eksekusi semuanya
            # 1) jadikan tool_calls dict yang valid untuk chat
            tool_calls_dicts = self._as_chat_tool_calls_dicts(calls)

            # 2) eksekusi masing-masing call
            tool_messages = []
            for td in tool_calls_dicts:
                fn = td["function"]["name"]
                args_str = td["function"]["arguments"] or "{}"
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {"_raw": args_str}

                result = await self._maybe_await(tool_executor(fn, args))

                # tool result
                if not isinstance(result, dict):
                    result = {"status": "success", "message": "ok", "data": result}
                result.setdefault("status", "success")
                result.setdefault("message", "ok")

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": td["id"],
                        "name": fn,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

            # 3) tambahkan satu pesan assistant (tool_calls) + beberapa pesan tool
            chat_msgs = [
                *chat_msgs,
                {"role": "assistant", "content": None, "tool_calls": tool_calls_dicts},
                *tool_messages,
            ]

            logger.info(
                "Roundtrip hop=%s prefer=%s (calls=%d)", hop, prefer, len(calls)
            )

        return self._ret(
            "error", f"Max hops tercapai ({max_hops}) tanpa final response (chat)."
        )

    # ======================================================================
    # --------------------------- FALLBACK TEXT -----------------------------
    # ======================================================================

    async def generate_text(
        self,
        messages_or_input: Union[str, List[Dict[str, Any]]],
        *,
        prefer: str = "chat",
        json_schema: Optional[Dict[str, Any]] = None,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """
        Hasilkan teks dengan fallback:
        - prefer "chat": coba Chat Completions (JSON Schema bila diberikan),
          fallback ke Responses (Pydantic bila diberikan).
        - prefer "responses": kebalikan.

        Args:
            messages_or_input: jika str → dipasangkan sebagai user content;
                               jika list → langsung dipakai.
            prefer: "chat"|"responses"
            json_schema: schema untuk Chat Completions (opsional)
            pydantic_model: model Pydantic untuk Responses (opsional)
        """
        # Normalisasi input ke dua bentuk:
        if isinstance(messages_or_input, str):
            chat_messages = [{"role": "user", "content": messages_or_input}]
            resp_input = messages_or_input
        else:
            chat_messages = messages_or_input
            resp_input = messages_or_input

        logger.info(
            "GenerateText (prefer=%s) | chat_schema=%s | pyd_model=%s",
            prefer,
            bool(json_schema),
            getattr(pydantic_model, "__name__", None),
        )

        async def _via_chat() -> Dict[str, Any]:
            return await self.chat_completions_text(
                chat_messages, json_schema=json_schema
            )

        async def _via_responses() -> Dict[str, Any]:
            return await self.responses_text(resp_input, pydantic_model=pydantic_model)

        first, second = (
            (_via_chat, _via_responses)
            if prefer == "chat"
            else (_via_responses, _via_chat)
        )

        r1 = await first()
        if r1.get("status") == "success":
            return r1

        logger.warning("GenerateText jalur pertama gagal → fallback.")
        r2 = await second()
        if r2.get("status") == "success":
            r2["message"] = f"{r2.get('message')} (fallback)"
            return r2

        # Keduanya gagal
        return self._ret(
            "error",
            f"Gagal generate teks via kedua jalur. "
            f"chat_err={r1.get('message')}; resp_err={r2.get('message')}",
        )

    # ======================================================================
    # -------------------------- PARSING UTIL -------------------------------
    # ======================================================================

    async def parse_text_with_pydantic(
        self,
        text: str,
        *,
        model: Type[BaseModel],
    ) -> Dict[str, Any]:
        """
        Validasi teks JSON menjadi Pydantic model (fallback parsing manual).
        Cocok sebagai langkah terakhir bila structured output sudah diminta.

        Args:
            text: string JSON.
            model: kelas Pydantic target.

        Returns:
            Dict konsisten {status, message, data?}
        """
        logger.info("Parse manual dengan Pydantic: %s", model.__name__)
        try:
            obj = model.model_validate_json(text)  # type: ignore[attr-defined]
            return self._ret(
                "success",
                "Berhasil mem-parse teks ke Pydantic.",
                data=obj.model_dump(),  # type: ignore[attr-defined]
            )
        except ValidationError as ve:
            logger.error("Gagal parse Pydantic: %s", ve)
            return self._ret("error", f"Validasi gagal: {ve.errors()}")
        except Exception as e:
            logger.error("Kesalahan parse tak terduga: %s", e, exc_info=True)
            return self._ret("error", self._friendly_error(e))
