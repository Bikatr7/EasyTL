## built-in libraries
import typing

## third party libraries
from google.generativeai import GenerationConfig
from google.generativeai.types import GenerateContentResponse, AsyncGenerateContentResponse

import google.generativeai as genai


## dummy values from my production code
_default_translation_instructions:str = "Format the response as JSON parseable string. It should have 2 keys, one for input titled input, and one called output, which is the translation."
_default_model:str = "gemini-1.5-pro-latest"

_system_message = _default_translation_instructions

_model:str = _default_model
_temperature:float = 0.5
_top_p:float = 0.9
_top_k:int = 40
_candidate_count:int = 1
_stream:bool = False
_stop_sequences:typing.List[str] | None = None
_max_output_tokens:int | None = None

_client:genai.GenerativeModel
_generation_config:GenerationConfig

_decorator_to_use:typing.Union[typing.Callable, None] = None

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

genai.configure(api_key="API_KEY")

_client = genai.GenerativeModel(model_name=_model,
                                safety_settings=_safety_settings,
                                system_instruction=_system_message)

_generation_config = GenerationConfig(candidate_count=_candidate_count,
                                                           stop_sequences=_stop_sequences,
                                                           max_output_tokens=_max_output_tokens,
                                                            temperature=_temperature,
                                                            top_p=_top_p,
                                                            top_k=_top_k,
                                                            response_mime_type="application/json")

print(_client.generate_content(
    "Hello, world!",generation_config=_generation_config
).text)