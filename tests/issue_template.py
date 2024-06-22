## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing

## third party libraries
from google.generativeai import GenerationConfig
from google.generativeai.types import GenerateContentResponse, AsyncGenerateContentResponse
import google.generativeai as genai

## Dummy values from production code
_default_translation_instructions: str = "Translate this to German. Format the response as JSON parseable string."
_default_model: str = "gemini-1.5-pro-latest"

_system_message = _default_translation_instructions

_model: str = _default_model
_temperature: float = 0.5
_top_p: float = 0.9
_top_k: int = 40
_candidate_count: int = 1
_stream: bool = False
_stop_sequences: typing.List[str] | None = None
_max_output_tokens: int | None = None

_client: genai.GenerativeModel
_generation_config: GenerationConfig

_decorator_to_use: typing.Union[typing.Callable, None] = None

_safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

## with open("gemini.txt", "r", encoding="utf-8") as f:
##      api_key = f.read().strip()
api_key = "YOUR_API_KEY"
genai.configure(api_key=api_key)

## Instructing the model to translate the input to German as JSON, without detailed schema
non_specific_client = genai.GenerativeModel(
    model_name=_model,
    safety_settings=_safety_settings,
    system_instruction="Translate this to German. Format the response as JSON parseable string."
)

## Instructing the model to translate the input to German as JSON, with detailed schema
_client = genai.GenerativeModel(
    model_name=_model,
    safety_settings=_safety_settings,
    system_instruction="Translate this to German. Format the response as JSON parseable string. It must have 2 keys, one for input titled input, and one called output, which is the translation."
)

_generation_config = GenerationConfig(
    candidate_count=_candidate_count,
    stop_sequences=_stop_sequences,
    max_output_tokens=_max_output_tokens,
    temperature=_temperature,
    top_p=_top_p,
    top_k=_top_k,
    response_mime_type="application/json",
    response_schema={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The original text that was translated."
            },
            "output": {
                "type": "string",
                "description": "The translated text."
            }
        },
        "required": ["input", "output"],
    }
)

## Inconsistent results, schema is not being followed
try:
    response = non_specific_client.generate_content(
        "Hello, world!", generation_config=_generation_config
    )
    print(response.text)
except Exception as e:
    print(f"Error with non-specific client: {e}")

## Consistent results, schema is being followed
try:
    response = _client.generate_content(
        "Hello, world!", generation_config=_generation_config
    )
    print(response.text)
except Exception as e:
    print(f"Error with specific client: {e}")

## Clarification question
## Is it intended behavior that the system instruction has to detail the schema? If so, what's the point of the response_schema parameter in the GenerationConfig class? It seems like a waste of tokens.