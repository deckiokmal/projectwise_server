from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

dekallm_url: str = "https://dekallm.cloudeka.ai/v1/"
clien = OpenAI(api_key = os.getenv("DEKA_LLM_API_KEY"),
                base_url= dekallm_url)

# api_key: str = str(os.getenv("DEKA_LLM_API_KEY"))
# client = OpenAI(base_url=dekallm_url, api_key=api_key)

system_prompt="kamu adalah seorang kapiten."
message="buat joke tentang kapiten."
response = aiModel.responses.create(
                model= "qwen/qwen25-72b-instruct",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            )

# print(response.output_text)
reply = response.choices[0].message.content
print(reply)

# =============================================================
# TODO: responses not work
# =============================================================
response = client.responses.create(
    model="qwen/qwen25-72b-instruct",
    input="tell me a joke about cat"
)

print(response.output_text)

r"""
Traceback (most recent call last):
  File "d:\1. PROJECTS\projectwise_quart\deka_llm.py", line 13, in <module>
    response = client.responses.create(
        model="qwen/qwen25-72b-instruct",
    ...<2 lines>...
        ]
    )
  File "D:\1. PROJECTS\projectwise_quart\.venv\Lib\site-packages\openai\resources\responses\responses.py", line 795, in create
    return self._post(
           ~~~~~~~~~~^
        "/responses",
        ^^^^^^^^^^^^^
    ...<38 lines>...
        stream_cls=Stream[ResponseStreamEvent],
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "D:\1. PROJECTS\projectwise_quart\.venv\Lib\site-packages\openai\_base_client.py", line 1259, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\1. PROJECTS\projectwise_quart\.venv\Lib\site-packages\openai\_base_client.py", line 1047, in request
    raise self._make_status_error_from_response(err.response) from None     
openai.NotFoundError: Error code: 404 - {'detail': 'Not Found'}
"""


# =============================================================
# TODO: chat completions work
# =============================================================
response_chat = clien.chat.completions.create(
    model="qwen/qwen25-72b-instruct", 
    messages=[
        {
            "role": "user", 
            "content": "halo !"
        }
    ]
)

print(response_chat.choices[0].message.content)

"""
{
    "id": "chatcmpl-4817b61762274e6585833e9abeb86194",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Halo! Apakah ada yang bisa saya bantu hari ini? Beri tahu saya apa yang Anda butuhkan atau tentang apa Anda ingin berbicara.",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "created": 1755677909,
    "model": "RedHatAI/Qwen2.5-72B-Instruct-quantized.w8a8",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 35,
        "prompt_tokens": 31,
        "total_tokens": 66,
        "completion_tokens_details": null,
        "prompt_tokens_details": null
    },
    "service_tier": null,
    "prompt_logprobs": null
}
"""


# =============================================================
# TODO: chat completions function call
# =============================================================

def get_current_weather():
    return "Cuaca mendung dengan suhu 100 derajat celcius"

tools = [
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
      },
    }
  }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = clien.chat.completions.create(
  model="qwen/qwen25-72b-instruct",
  messages=messages, # type: ignore
  tools=tools, # type: ignore
  tool_choice="auto"
)

print(completion)

"""
# prompt indikasi untuk melakukan tool call
# tool_choice="auto"
{
    "id": "chatcmpl-969dae258dd54b1e9305a9a782bb5f65",
    "choices": [
        {
            "finish_reason": "tool_calls",
            "index": 0,
            "message": {
                "content": null,
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": "{\"location\": \"Boston, MA\", \"unit\": \"fahrenheit\"}",
                            "name": "get_current_weather"
                        },
                        "id": "chatcmpl-tool-6c19640e620143e6b6e2bed4f5bedc5a",
                        "type": "function"
                    }
                ],
                "function_call": null
            }
        }
    ],
    "created": 1755678590,
    "model": "RedHatAI/Qwen2.5-72B-Instruct-quantized.w8a8",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 30,
        "prompt_tokens": 212,
        "total_tokens": 242,
        "completion_tokens_details": null,
        "prompt_tokens_details": null
    },
    "service_tier": null,
    "prompt_logprobs": null
}

# prompt indikasi untuk melakukan tool call
# tool_choice="none"
{
    "id": "chatcmpl-6cfa7d1e285546fba0cf16493a3f9b7e",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "I don't have real-time data access...,
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "created": 1755678674,
    "model": "RedHatAI/Qwen2.5-72B-Instruct-quantized.w8a8",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 100,
        "prompt_tokens": 38,
        "total_tokens": 138,
        "completion_tokens_details": null,
        "prompt_tokens_details": null
    },
    "service_tier": null,
    "prompt_logprobs": null
}

# prompt tidak terindikasi untuk melakukan tool call
# tool_choice="auto"
{
    "id": "chatcmpl-7138490f68074314b6a894db135063d4",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Makanan favorit di Indonesia...,
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "created": 1755678825,
    "model": "RedHatAI/Qwen2.5-72B-Instruct-quantized.w8a8",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 564,
        "prompt_tokens": 215,
        "total_tokens": 779,
        "completion_tokens_details": null,
        "prompt_tokens_details": null
    },
    "service_tier": null,
    "prompt_logprobs": null
}
"""



# Test embedding

"""
kenapa menggunakan openai:
    1. masalah token rate limit yang terbatas untuk mengelola dokumen besar
    2. 

- butuh model dengan token limit yang lebih besar. contohnya model: Qwen/Qwen2.5-7B-Instruct-1M
- embedding model: gemma

"""