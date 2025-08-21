# scripts/run_llm_test.py
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List
from mcp_server.utils.llm_chains import LLMChains


# Test 1 putaran tool call dengan chat.completions
async def test():
    llm = LLMChains(model="gpt-4o-mini")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Cuaca Palembang sekarang pakai Celcius?"},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["c", "f"]},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    fc = await llm.function_call(messages, tools=tools, prefer="chat")  # noqa: F704
    if fc["status"] != "success" or not fc.get("calls"):
        raise RuntimeError(fc.get("message", "Function call gagal."))

    call = fc["calls"][0]
    func_name = call["name"]
    func_args = call["arguments"]
    tool_call_id = call.get("id")

    # Executor contoh (mock)
    def get_weather(city: str, unit: str = "c"):
        return {"temp_c": 31, "condition": "Cloudy"}

    result = get_weather(**func_args)

    assistant_msg = fc["raw"].choices[0].message
    messages2 = [
        *messages,
        {
            "role": "assistant",
            "content": assistant_msg.content,
            "tool_calls": assistant_msg.tool_calls,
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": func_name,
            "content": json.dumps(result, ensure_ascii=False),
        },
    ]

    final_resp = await llm.chat_completions_text(messages2)

    # Tampilkan ringkas di terminal
    print("STATUS :", final_resp.get("status"))
    print("MESSAGE  :", final_resp.get("message"))
    print("OUTPUT :", final_resp.get("data") or final_resp)
    print("RAW : ", final_resp.get("raw"))
    print("META : ", final_resp.get("meta"))


# ──────────────────────────────────────────────────────────────
# 5 MOCK TOOLS (pure Python)
# ──────────────────────────────────────────────────────────────
def get_weather(city: str, unit: str = "c") -> Dict[str, Any]:
    # Simulasi: data statis
    temp_c = 31
    return {
        "city": city,
        "unit": unit,
        "temp_c": temp_c,
        "temp_f": round(temp_c * 9 / 5 + 32, 1),
        "condition": "Cloudy",
        "source": "mock_weather",
    }

def get_time(city: str) -> Dict[str, Any]:
    # Simulasi: pakai waktu lokal server sebagai contoh
    return {
        "city": city,
        "iso_time": datetime.now().isoformat(timespec="seconds"),
        "timezone_assumed": "server_local",
        "source": "mock_time",
    }

def get_currency_rate(base: str, target: str) -> Dict[str, Any]:
    # Simulasi: nilai contoh
    sample = {
        ("USD", "IDR"): 15400.0,
        ("IDR", "USD"): 1 / 15400.0,
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
    }
    rate = sample.get((base.upper(), target.upper()), 1.0)
    return {"base": base.upper(), "target": target.upper(), "rate": rate, "source": "mock_fx"}

def get_air_quality(city: str) -> Dict[str, Any]:
    # Simulasi: nilai AQI contoh
    return {
        "city": city,
        "aqi": 78,
        "category": "Moderate",
        "pm25": 22.5,
        "pm10": 40.1,
        "source": "mock_aqi",
    }

def get_local_news(city: str, limit: int = 3) -> Dict[str, Any]:
    # Simulasi: headline statis
    headlines = [
        f"{city}: Peningkatan Infrastruktur Jaringan",
        f"{city}: Festival Kuliner Minggu Ini",
        f"{city}: Update Lalu Lintas Pagi Hari",
    ][:limit]
    return {"city": city, "headlines": headlines, "source": "mock_news"}


# Mapping nama tool → fungsi eksekusi Python
TOOL_EXECUTORS = {
    "get_weather": get_weather,
    "get_time": get_time,
    "get_currency_rate": get_currency_rate,
    "get_air_quality": get_air_quality,
    "get_local_news": get_local_news,
}


# ──────────────────────────────────────────────────────────────
# Definisi 5 tools dalam skema OpenAI "function calling"
# ──────────────────────────────────────────────────────────────
TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_currency_rate",
            "description": "Get latest currency exchange rate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["base", "target"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality",
            "description": "Get the air quality index for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_news",
            "description": "Get top local news headlines for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────
# 1) Iterasi multiple tool-call hingga selesai
#    - Mengumpulkan seluruh hasil tools dalam satu aggregator
#    - Mengembalikan messages + agregat untuk tahap finalisasi
# ──────────────────────────────────────────────────────────────
async def run_multi_tool_iteration(
    question: str,
    city: str = "Palembang",
    base_ccy: str = "USD",
    target_ccy: str = "IDR",
    news_limit: int = 3,
    model: str = "gpt-4o-mini",
    max_steps: int = 8,
) -> Dict[str, Any]:
    """
    Jalankan loop function-calling: model meminta tool → eksekusi mock → injeksi balik.
    Berhenti bila:
      - Tidak ada tool call lagi, atau
      - Mencapai max_steps.
    """
    llm = LLMChains(model=model)

    # Instruksi agar model menggabungkan data lintas-tool
    system_prompt = (
        "You are a helpful assistant. "
        "Kamu boleh memanggil beberapa fungsi untuk mengumpulkan data, "
        "lalu gabungkan menjadi 1 jawaban final yang ringkas dan akurat. "
        "Jika butuh kurs, gunakan get_currency_rate. Jika butuh AQI, gunakan get_air_quality, dst."
    )
    user_prompt = (
        f"Tolong rangkum kondisi kota {city} sekarang: cuaca (Celcius), waktu lokal,"
        f" kualitas udara, kurs {base_ccy}/{target_ccy}, dan 3 headline berita lokal."
        " Berikan jawaban akhir yang singkat, akurat, dan actionable."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Data tambahan agar model punya konteks default argumen
    context_hint = {
        "city": city,
        "base_ccy": base_ccy,
        "target_ccy": target_ccy,
        "news_limit": news_limit,
    }

    aggregator: Dict[str, Any] = {
        "weather": None,
        "time": None,
        "fx": None,
        "aqi": None,
        "news": None,
        "_meta": {"steps": 0, "errors": []},
        "_hint": context_hint,
    }

    for step in range(1, max_steps + 1):
        fc = await llm.function_call(messages, tools=TOOLS_SCHEMA, prefer="chat")
        aggregator["_meta"]["steps"] = step

        if fc.get("status") != "success":
            aggregator["_meta"]["errors"].append(
                {"step": step, "where": "function_call", "message": fc.get("message")}
            )
            break

        # Ambil isi assistant raw
        assistant_msg = fc["raw"].choices[0].message
        tool_calls = getattr(assistant_msg, "tool_calls", None)

        # Jika tidak ada tool_call lagi → selesai iterasi
        if not tool_calls and not fc.get("calls"):
            # Simpan teks asistennya jika ada
            if assistant_msg.content:
                messages.append({"role": "assistant", "content": assistant_msg.content})
            break

        # Pastikan assistant message (dengan tool_calls) masuk ke messages SATU KALI sebelum tool outputs
        messages.append(
            {
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": assistant_msg.tool_calls,
            }
        )

        # Eksekusi SEMUA panggilan tool yang diminta model di step ini
        # Gunakan fc["calls"] (LLMChains flatten) sebagai sumber argumen yang sudah diparse
        calls: List[Dict[str, Any]] = fc.get("calls", [])
        for call in calls:
            tool_name = call["name"]
            args = call["arguments"] or {}
            tool_call_id = call.get("id")

            # Fallback pengisian argumen minimal (bila model lupa)
            if tool_name in {"get_weather", "get_time", "get_air_quality", "get_local_news"}:
                args.setdefault("city", city)
            if tool_name == "get_currency_rate":
                args.setdefault("base", base_ccy)
                args.setdefault("target", target_ccy)
            if tool_name == "get_local_news":
                args.setdefault("limit", news_limit)
            if tool_name == "get_weather":
                args.setdefault("unit", "c")

            # Eksekusi
            try:
                pyfunc = TOOL_EXECUTORS[tool_name]
                result = pyfunc(**args)
            except Exception as e:
                result = {"error": str(e), "tool": tool_name}

            # Masukkan ke aggregator
            if tool_name == "get_weather":
                aggregator["weather"] = result
            elif tool_name == "get_time":
                aggregator["time"] = result
            elif tool_name == "get_currency_rate":
                aggregator["fx"] = result
            elif tool_name == "get_air_quality":
                aggregator["aqi"] = result
            elif tool_name == "get_local_news":
                aggregator["news"] = result

            # Injeksi hasil tool ke dalam messages
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return {
        "status": "success",
        "messages": messages,
        "aggregate": aggregator,
        "model": model,
    }


# ──────────────────────────────────────────────────────────────
# 2) Hanya menghasilkan jawaban final (tanpa tool-call baru)
#    - Menggunakan messages/aggregate yang sudah terbentuk
# ──────────────────────────────────────────────────────────────
async def generate_final_answer(
    messages: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Menyusun jawaban final berbasis data tools yang SUDAH tersedia.
    Tidak memanggil tool apapun → gunakan chat_completions_text.
    """
    llm = LLMChains(model=model)

    synthesis_instruction = (
        "Tugasmu sekarang: SUSUN SATU JAWABAN FINAL RINGKAS dari data berikut. "
        "Jangan memanggil fungsi lagi, cukup gunakan data yang ada. "
        "Formatkan dalam 4 bagian singkat: Cuaca, Waktu, Kualitas Udara, Kurs, Headline. "
        "Gunakan bullet singkat, satu paragraf ringkasan di akhir (actionable). "
        "Jika ada bagian tidak tersedia, singkatkan dan beri catatan 'n/a'."
    )

    # Kita sisipkan konteks aggregate sebagai pesan system tambahan
    messages_for_final = [
        *messages,
        {
            "role": "system",
            "content": synthesis_instruction
            + "\n\nDATA_AGGREGATE:\n```json\n"
            + json.dumps(aggregate, ensure_ascii=False, indent=2)
            + "\n```",
        },
        {"role": "user", "content": "Tulis jawaban final sekarang."},
    ]

    final_resp = await llm.chat_completions_text(messages_for_final)
    return {
        "status": final_resp.get("status"),
        "model": final_resp.get("model"),
        "data": final_resp.get("data"),
    }


# ──────────────────────────────────────────────────────────────
# Runner (terminal)
# ──────────────────────────────────────────────────────────────
async def main():
    # Konfigurasi contoh
    question = (
        "Tolong rangkum kondisi untuk pengambilan keputusan cepat hari ini."
    )
    city = "Palembang"
    base_ccy = "USD"
    target_ccy = "IDR"
    news_limit = 3

    # 1) Iterasi multi-tool (kumpulkan semua hasil dulu)
    r_iter = await run_multi_tool_iteration(
        question=question,
        city=city,
        base_ccy=base_ccy,
        target_ccy=target_ccy,
        news_limit=news_limit,
        model="gpt-4o-mini",
        max_steps=8,
    )

    print("\n=== AGGREGATE (ringkas) ===")
    agg_preview = {
        "weather": r_iter["aggregate"]["weather"],
        "time": r_iter["aggregate"]["time"],
        "fx": r_iter["aggregate"]["fx"],
        "aqi": r_iter["aggregate"]["aqi"],
        "news": r_iter["aggregate"]["news"],
        "_meta": r_iter["aggregate"]["_meta"],
    }
    print(json.dumps(agg_preview, ensure_ascii=False, indent=2))

    # 2) Finalisasi jawaban (tanpa tool-call baru)
    r_final = await generate_final_answer(
        messages=r_iter["messages"],
        aggregate=r_iter["aggregate"],
        model="gpt-4o-mini",
    )

    print("\n=== FINAL ANSWER ===")
    print("STATUS:", r_final["status"])
    print("MODEL :", r_final["model"])
    print("OUTPUT:\n", r_final["data"])
    

# executor backend Anda (boleh async/sync)
async def tool_executor(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "get_weather":
        city = args.get("city", "Palembang")
        unit = args.get("unit", "c")
        # Panggil API/cuaca nyata di sini...
        return {"city": city, "unit": unit, "temp_c": 31, "condition": "Cloudy"}
    return {"error": f"unknown tool {name}"}

async def ask_weather_roundtrip(city: str):
    llm = LLMChains(model="gpt-4o-mini")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Cuaca {city} sekarang pakai Celcius?"}
    ]

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["city"],
                "additionalProperties": False
            }
        }
    }]

    # Roundtrip multi-hop sampai final
    result = await llm.run_function_call_roundtrip(
        messages,
        tools=tools,
        tool_executor=tool_executor,
        prefer="chat",          # ← produksi: chat
        max_hops=4,
    )
    return result  # dict {status, message, data, ...}


# Test multiple tool call dengan chat.completions
if __name__ == "__main__":
    asyncio.run(main())
