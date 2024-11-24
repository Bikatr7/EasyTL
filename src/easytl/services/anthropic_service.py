## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio

## third-party imports
from anthropic import Anthropic, AsyncAnthropic

## custom modules
from ..exceptions import EasyTLException
from ..classes import ModelTranslationMessage, NotGiven, NOT_GIVEN, AnthropicMessage

from ..util.util import _is_iterable_of_strings, _convert_iterable_to_str, _estimate_cost
from ..util.constants import VALID_JSON_ANTHROPIC_MODELS


class AnthropicService:

    _default_model:str = "claude-3-haiku-20240307"
    _default_translation_instructions:str = "Please translate the following text into English."

    _system:str | None   = _default_translation_instructions 

    _model:str = _default_model
    _temperature:float | NotGiven = NOT_GIVEN
    _top_p:float | NotGiven = NOT_GIVEN
    _top_k:int | NotGiven = NOT_GIVEN
    _stream:bool = False
    _stop_sequences:typing.List[str] | NotGiven = NOT_GIVEN
    _max_tokens:int | NotGiven = NOT_GIVEN

    _semaphore_value:int = 5
    _semaphore:asyncio.Semaphore = asyncio.Semaphore(_semaphore_value)

    _sync_client = Anthropic(api_key="DummyKey")
    _async_client = AsyncAnthropic(api_key="DummyKey")

    _rate_limit_delay:float | None = None

    _decorator_to_use:typing.Union[typing.Callable, None] = None

    _json_mode:bool = False
    _response_schema:typing.Mapping[str, typing.Any] | None = None
    
    _json_tool = {
        "name": "format_to_json",
        "description": "Formats text into json. This is required.",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The text you were given to translate"
                },
                "output": {
                    "type": "string",
                    "description": "The translated text"
                }
            },
            "required": ["input", "output"]
        }
    }

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
                        system:str | None = _default_translation_instructions,
                        temperature:float | NotGiven = NOT_GIVEN,
                        top_p:float | NotGiven = NOT_GIVEN,
                        top_k:int | NotGiven = NOT_GIVEN,
                        stream:bool = False,
                        stop_sequences:typing.List[str] | NotGiven = NOT_GIVEN,
                        max_tokens:int | NotGiven = NOT_GIVEN,
                        decorator:typing.Union[typing.Callable, None]=None,
                        semaphore:int | None=None,
                        rate_limit_delay:float | None=None,
                        json_mode:bool=False,
                        response_schema:typing.Mapping[str, typing.Any] | None = None
                        ) -> None:
    
            """
    
            Sets the attributes for the Anthropic Service.
    
            """
    
            AnthropicService._model = model
            AnthropicService._system = system
            AnthropicService._temperature = temperature
            AnthropicService._top_p = top_p
            AnthropicService._top_k = top_k
            AnthropicService._stream = stream
            AnthropicService._stop_sequences = stop_sequences
            AnthropicService._max_tokens = max_tokens

            AnthropicService._decorator_to_use = decorator

            AnthropicService._rate_limit_delay = rate_limit_delay

            AnthropicService._json_mode = json_mode
            AnthropicService._response_schema = response_schema

            AnthropicService._json_tool['input_schema'] = AnthropicService._response_schema

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
    
##-------------------start-of-_translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _translate_text(translation_instructions: typing.Optional[str],
                                translation_prompt: ModelTranslationMessage
                                ) -> AnthropicMessage | typing.Iterator[AnthropicMessage]:
        
        """
        
        Synchronously translates the text using the Anthropic API.

        Parameters:
        translation_instructions (str) : The instructions to use for the translation.
        translation_prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (AnthropicMessage) : The response from the API.

        """
        
        if(translation_instructions is None):
            translation_instructions = AnthropicService._default_translation_instructions

        if(AnthropicService._decorator_to_use is None):
            return AnthropicService.__translate_text(translation_instructions, translation_prompt)

        decorated_function = AnthropicService._decorator_to_use(AnthropicService.__translate_text)
        return decorated_function(translation_instructions, translation_prompt)
    
##-------------------start-of-_translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def _translate_text_async(translation_instructions: typing.Optional[str],
                                translation_prompt: ModelTranslationMessage
                                ) -> AnthropicMessage | typing.AsyncIterator[AnthropicMessage]:
        
        """

        Asynchronously translates the text using the Anthropic API.

        Parameters:
        translation_instructions (str) : The instructions to use for the translation.
        translation_prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (AnthropicMessage | AsyncIterator[AnthropicMessage]) : The response from the API.

        """
        
        if(translation_instructions is None):
            translation_instructions = AnthropicService._default_translation_instructions

        if(AnthropicService._decorator_to_use is None):
            return await AnthropicService.__translate_text_async(translation_instructions, translation_prompt)
        
        decorated_function = AnthropicService._decorator_to_use(AnthropicService.__translate_text_async)
        return await decorated_function(translation_instructions, translation_prompt)

##-------------------start-of-_translate_message()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __translate_text(instructions:str, prompt:ModelTranslationMessage) -> AnthropicMessage | typing.Iterator[AnthropicMessage]:

        """

        Synchronously translates the text using the Anthropic API.

        Parameters:
        instructions (str) : The instructions to use for the translation.
        prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (AnthropicMessage) : The response from the API.

        """
        
        attributes = ["temperature", "top_p", "top_k", "stream", "stop_sequences", "max_tokens"]
        message_args = {
            "model": AnthropicService._model,
            "system": instructions,
            "messages": [prompt.to_dict()],
            "stream": AnthropicService._stream,
            **{attr: getattr(AnthropicService, f"_{attr}") for attr in attributes if getattr(AnthropicService, f"_{attr}") != NOT_GIVEN and attr != "stream"}
        }
        
        ## Special case for max_tokens
        message_args["max_tokens"] = message_args.get("max_tokens", 4096)
        
        if(AnthropicService._json_mode and AnthropicService._model in VALID_JSON_ANTHROPIC_MODELS):
            message_args.update({
                "tools": [AnthropicService._json_tool],
                "tool_choice": {"type": "tool", "name": "format_to_json"}
            })
        
        response = AnthropicService._sync_client.messages.create(**message_args)
        return response
    
##-------------------start-of- __translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def __translate_text_async(instructions:str, prompt:ModelTranslationMessage) -> AnthropicMessage | typing.AsyncIterator[AnthropicMessage]:

        """

        Asynchronously translates the text using the Anthropic API.

        Parameters:
        instruction (str) : The instructions to use for the translation.
        prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (AnthropicMessage) : The response from the API.

        """

        async with AnthropicService._semaphore:

            if(AnthropicService._rate_limit_delay is not None):
                await asyncio.sleep(AnthropicService._rate_limit_delay)

            attributes = ["temperature", "top_p", "top_k", "stream", "stop_sequences", "max_tokens"]
            message_args = {
                "model": AnthropicService._model,
                "system": instructions,
                "messages": [prompt.to_dict()],
                "stream": AnthropicService._stream,
                **{attr: getattr(AnthropicService, f"_{attr}") for attr in attributes if getattr(AnthropicService, f"_{attr}") != NOT_GIVEN and attr != "stream"}
            }
            
            ## Special case for max_tokens
            message_args["max_tokens"] = message_args.get("max_tokens", 4096)
            
            if(AnthropicService._json_mode and AnthropicService._model in VALID_JSON_ANTHROPIC_MODELS):
                message_args.update({
                    "tools": [AnthropicService._json_tool],
                    "tool_choice": {"type": "tool", "name": "format_to_json"}
                })

            response = await AnthropicService._async_client.messages.create(**message_args)

            return response
    
##-------------------start-of-test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _test_api_key_validity() -> typing.Tuple[bool, typing.Union[Exception, None]]:

        """

        Tests the validity of the API key.

        Returns:
        validity (bool) : True if the API key is valid, False if it is not.
        e (Exception) : The exception that was raised, if any.

        """

        _validity = False

        try:

            AnthropicService._sync_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": "Respond to this with 1"},
                ]

            )

            _validity = True

            return _validity, None

        except Exception as _e:

            return _validity, _e
        
    
##-------------------start-of-_calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _calculate_cost(text:str | ModelTranslationMessage | typing.Iterable, translation_instructions:str | None, model:str | None) -> typing.Tuple[int, float, str]:

        """

        Calculates the cost of the translation.

        Parameters:
        text (string | iterable) : The text to translate.
        translation_instructions (string) : The instructions to use for the translation.
        model (string) : The model to use for the translation.

        Returns:
        num_tokens (int) : The number of tokens.
        cost (float) : The cost of the translation.
        model (string) : The model used for the translation.

        """

        cost_modifier = 1.0

        if(translation_instructions is None):
            translation_instructions = AnthropicService._default_translation_instructions

        if(isinstance(text, typing.Iterable)):

            if(not isinstance(text,str) and not _is_iterable_of_strings(text)):
                raise ValueError("The text must be a string or an iterable of strings.")
            
            if(isinstance(text, ModelTranslationMessage)):
                ## since instructions are paired with the text, we need to repeat the instructions for index
                ## this only works if the text is pre-built ModelTranslationMessage objects
                ## otherwise, the instructions will be repeated for each item in the iterable which is not the intended behavior
                translation_instructions = translation_instructions * len(text) # type: ignore

            else:
                ## otherwise, we can really only estimate.
                cost_modifier = 2.5

            text = _convert_iterable_to_str(text)

        if(isinstance(translation_instructions, typing.Iterable)):
            translation_instructions = _convert_iterable_to_str(translation_instructions)

        if(model is None):
            model = AnthropicService._default_model

        ## not exactly how the text will be formatted, but it's close enough for the purposes of estimating the cost as tokens should be the same
        total_text_to_estimate = f"{translation_instructions}\n{text}"
        
        _num_tokens, _cost, _ = _estimate_cost(total_text_to_estimate, model)

        _cost = _cost * cost_modifier

        return _num_tokens, _cost, model   