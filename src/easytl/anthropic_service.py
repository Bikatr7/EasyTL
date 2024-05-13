## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio

## third-party imports
from anthropic import Anthropic, AsyncAnthropic

## custom modules
from .exceptions import EasyTLException
from .classes import ModelTranslationMessage
from .util import ALLOWED_ANTHROPIC_MODELS, VALID_JSON_ANTHROPIC_MODELS

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

##-------------------start-of-set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_api_key(api_key:str) -> None:

        """

        Sets the API key for the Anthropic Clients.

        Parameters:
        api_key (string) : The API key to set.

        """

        AnthropicService._async_client.api_key = api_key
        AnthropicService._sync_client.api_key = api_key

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def _set_attributes(model:str = _default_model,
                        temperature:float = 0.3,
                        top_p:float = 1.0,
                        top_k:int = 40,
                        stream:bool = False,
                        stop_sequences:typing.List[str] | None = None,
                        max_tokens:int | None = None,
                        decorator:typing.Union[typing.Callable, None]=None,
                        logging_directory:str | None=None,
                        semaphore:int | None=None,
                        rate_limit_delay:float | None=None,
                        json_mode:bool=False
                        ) -> None:
    
            """
    
            Sets the attributes for the Anthropic Service.
    
            """
    
            AnthropicService._model = model
            AnthropicService._temperature = temperature
            AnthropicService._top_p = top_p
            AnthropicService._top_k = top_k
            AnthropicService._stream = stream
            AnthropicService._stop_sequences = stop_sequences
            AnthropicService._max_tokens = max_tokens

            AnthropicService._decorator_to_use = decorator

            AnthropicService._log_directory = logging_directory

            AnthropicService._rate_limit_delay = rate_limit_delay

            AnthropicService._json_mode = json_mode

            ## if a decorator is used, we want to disable retries, otherwise set it to the default value which is 2
            if(AnthropicService._decorator_to_use is not None):
                AnthropicService._sync_client.max_retries = 0
                AnthropicService._async_client.max_retries = 0
            else:
                AnthropicService._sync_client.max_retries = 2
                AnthropicService._async_client.max_retries = 2
            
            if(semaphore is not None):
                AnthropicService._semaphore_value = semaphore
                AnthropicService._semaphore = asyncio.Semaphore(AnthropicService._semaphore_value)

            if(AnthropicService._json_mode and AnthropicService._model in VALID_JSON_ANTHROPIC_MODELS):
                AnthropicService._default_translation_instructions = "Please translate the following text into English. Make sure to use the translate tool to return the text in a valid JSON format."

            elif(AnthropicService._json_mode):
                model_string = ", ".join(VALID_JSON_ANTHROPIC_MODELS)
                raise EasyTLException("JSON mode for Anthropic is only available for the following models: " + model_string)
            
            else:
                AnthropicService._default_translation_instructions = "Please translate the following text into English."

##-------------------start-of-_build_translation_batches()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _build_translation_batches(text: typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]]) -> typing.List[ModelTranslationMessage]:
        
        """
        
        Builds the translation batches for the Anthropic service.

        Parameters:
        text (string | iterable[string] | ModelTranslationMessage | iterable[ModelTranslationMessage]) : The text to translate.

        Returns:
        translation_batches (list[ModelTranslationMessage]) : The translation batches to send to the Anthropic service.

        """

        
        if(isinstance(text, str)):
            
            text = [ModelTranslationMessage(content=text)]

        elif(isinstance(text, ModelTranslationMessage)):
            text = [text]
        
        elif(isinstance(text, typing.Iterable)):
            text = [ModelTranslationMessage(content=item) if isinstance(item, str) else item for item in text]
        else:
            raise ValueError("Invalid type for text. Must either be a string, ModelTranslationMessage, or an iterable of strings/ModelTranslationMessage.")
        
        if(any(not isinstance(item, ModelTranslationMessage) for item in text)):
            raise ValueError("Invalid type in iterable. Must be either strings or ModelTranslationMessage objects.")
        
        return text
    

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