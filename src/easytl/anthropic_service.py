## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio

## third-party imports
from anthropic import Anthropic, AsyncAnthropic

class AnthropicService:

    _default_model:str = "claude-3-haiku-20240307"
    _default_translation_instructions:str = "Please translate the following text into English."

    _system:str = _default_translation_instructions

    _model:str = _default_model
    _temperature:float = 0.3
    _top_p:float = 1.0
    _top_k:int = 40
    _stream:bool = False
    _stop_sequences:typing.List[str] | None = None
    _max_tokens:int | None = None

    _semaphore_value:int = 5
    _semaphore:asyncio.Semaphore = asyncio.Semaphore(_semaphore_value)

    _sync_client = Anthropic(api_key="DummyKey")
    _async_client = AsyncAnthropic(api_key="DummyKey")

    _rate_limit_delay:float | None = None

    _decorator_to_use:typing.Union[typing.Callable, None] = None

    _log_directory:str | None = None

    _json_mode:bool = False

def main():

     with open ("./tests/anthropic.txt", "r", encoding="utf-8") as file:
         api_key = file.read().strip()

     client = Anthropic(
         api_key=api_key
     )

     message = client.messages.create(
         model="claude-3-haiku-20240307",
         max_tokens=100,
         temperature=0.0,
         system="You are a chatbot.",
         messages=[
             {"role": "user", "content": "Respond to this with 1"},
         ]
     )

     print(message.content[0].text)