## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio

## third-party libraries
from openai import AsyncOpenAI, OpenAI

## custom modules
from ..ld_helper import LDHelper
from ..classes import SystemTranslationMessage, ModelTranslationMessage, ChatCompletion, NOT_GIVEN, NotGiven
from ..decorators import _async_logging_decorator, _sync_logging_decorator
from ..exceptions import EasyTLException

from ..util.util import _convert_iterable_to_str, _estimate_cost, _is_iterable_of_strings
from ..util.constants import VALID_JSON_OPENAI_MODELS


class OpenAIService:

    _default_model:str = "gpt-4"
    _default_translation_instructions:SystemTranslationMessage = SystemTranslationMessage("Please translate the following text into English.")

    _system_message:typing.Optional[typing.Union[SystemTranslationMessage, str]] = _default_translation_instructions

    _model:str = _default_model
    _temperature:float | None | NotGiven = NOT_GIVEN
    _logit_bias:typing.Dict[str, int] | None | NotGiven = NOT_GIVEN
    _top_p:float | None | NotGiven = NOT_GIVEN
    _n:int | None | NotGiven = 1
    _stream:bool = False
    _stop:typing.List[str] | None | NotGiven = NOT_GIVEN
    _max_tokens:int | None | NotGiven = NOT_GIVEN
    _presence_penalty:float | None | NotGiven = NOT_GIVEN
    _frequency_penalty:float | None | NotGiven = NOT_GIVEN

    _semaphore_value:int = 5
    _semaphore:asyncio.Semaphore = asyncio.Semaphore(_semaphore_value)

    _sync_client = OpenAI(api_key="DummyKey")
    _async_client = AsyncOpenAI(api_key="DummyKey")

    _rate_limit_delay:float | None = None

    _decorator_to_use:typing.Union[typing.Callable, None] = None

    _log_directory:str | None = None

    _json_mode:bool = False

##-------------------start-of-set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_api_key(api_key:str) -> None:

        """

        Sets the API key for the OpenAI clients.

        Parameters:
        api_key (string) : The API key to set.

        """

        OpenAIService._async_client.api_key = api_key
        OpenAIService._sync_client.api_key = api_key

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def _set_attributes(model:str = _default_model,
                        temperature:float | None | NotGiven = NOT_GIVEN,
                        logit_bias:typing.Dict[str, int] | None | NotGiven = NOT_GIVEN,
                        top_p:float | None | NotGiven = NOT_GIVEN,
                        n:int | None | NotGiven = 1,
                        stream:bool = False,
                        stop:typing.List[str] | None | NotGiven = NOT_GIVEN,
                        max_tokens:int | None | NotGiven = NOT_GIVEN,
                        presence_penalty:float | None | NotGiven = NOT_GIVEN,
                        frequency_penalty:float | None | NotGiven = NOT_GIVEN,
                        decorator:typing.Union[typing.Callable, None]=None,
                        logging_directory:str | None=None,
                        semaphore:int | None=None,
                        rate_limit_delay:float | None=None,
                        json_mode:bool=False
                        ) -> None:
    
            """
    
            Sets the attributes for the OpenAI service.
    
            """
    
            OpenAIService._model = model
            OpenAIService._temperature = temperature
            OpenAIService._logit_bias = logit_bias
            OpenAIService._top_p = top_p
            OpenAIService._n = n
            OpenAIService._stream = stream
            OpenAIService._stop = stop
            OpenAIService._max_tokens = max_tokens
            OpenAIService._presence_penalty = presence_penalty
            OpenAIService._frequency_penalty = frequency_penalty

            OpenAIService._decorator_to_use = decorator

            OpenAIService._log_directory = logging_directory

            ## For Elucidate
            LDHelper.set_logging_directory_attributes(logging_directory, "OpenAIService")

            OpenAIService._rate_limit_delay = rate_limit_delay

            OpenAIService._json_mode = json_mode

            ## if a decorator is used, we want to disable retries, otherwise set it to the default value which is 2
            if(OpenAIService._decorator_to_use is not None):
                OpenAIService._sync_client.max_retries = 0
                OpenAIService._async_client.max_retries = 0
            else:
                OpenAIService._sync_client.max_retries = 2
                OpenAIService._async_client.max_retries = 2
            
            if(semaphore is not None):
                OpenAIService._semaphore_value = semaphore
                OpenAIService._semaphore = asyncio.Semaphore(OpenAIService._semaphore_value)

            if(OpenAIService._json_mode and OpenAIService._model in VALID_JSON_OPENAI_MODELS):
                OpenAIService._default_translation_instructions = SystemTranslationMessage("Please translate the following text into English. Make sure to return the translated text in JSON format.")

            elif(OpenAIService._json_mode):
                model_string = ", ".join(VALID_JSON_OPENAI_MODELS)
                raise EasyTLException("JSON mode for OpenAI is only available for the following models: " + model_string)
            
            else:
                OpenAIService._default_translation_instructions = SystemTranslationMessage("Please translate the following text into English.")

##-------------------start-of-_build_translation_batches()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _build_translation_batches(text: typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]],
                                instructions: typing.Optional[typing.Union[str, SystemTranslationMessage]] = None) -> typing.List[typing.Tuple[ModelTranslationMessage, SystemTranslationMessage]]:
        
        """
        
        Builds the translation batches for the OpenAI service.

        Parameters:
        text (string | iterable[string] | ModelTranslationMessage | iterable[ModelTranslationMessage]) : The text to translate.
        instructions (string | SystemTranslationMessage) : The instructions to use for the translation.

        Returns:
        translation_batches (list[tuple[ModelTranslationMessage, SystemTranslationMessage]]) : The translation batches.

        """

        
        if(isinstance(instructions, str)):
            instructions = SystemTranslationMessage(instructions)

        elif(not isinstance(instructions, SystemTranslationMessage)):
            raise ValueError("Invalid type for instructions. Must either be a string or a pre-built SystemTranslationMessage object.")

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
        
        return [(item, instructions) for item in text]

##-------------------start-of-_translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    @_sync_logging_decorator
    def _translate_text(translation_instructions: typing.Optional[SystemTranslationMessage],
                                translation_prompt: ModelTranslationMessage
                                ) -> ChatCompletion:
        
        """
        
        Synchronously translates the text using the OpenAI API.

        Parameters:
        translation_instructions (SystemTranslationMessage) : The instructions to use for the translation.
        translation_prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (ChatCompletion) : The response from the API.

        """
        
        if(translation_instructions is None):
            translation_instructions = OpenAIService._default_translation_instructions

        if(OpenAIService._decorator_to_use is None):
            return OpenAIService.__translate_text(translation_instructions, translation_prompt)

        decorated_function = OpenAIService._decorator_to_use(OpenAIService.__translate_text)
        return decorated_function(translation_instructions, translation_prompt)
    
##-------------------start-of-_translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    @_async_logging_decorator
    async def _translate_text_async(translation_instructions: typing.Optional[SystemTranslationMessage],
                                translation_prompt: ModelTranslationMessage
                                ) -> ChatCompletion:
        
        """

        Asynchronously translates the text using the OpenAI API.

        Parameters:
        translation_instructions (SystemTranslationMessage) : The instructions to use for the translation.
        translation_prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (ChatCompletion) : The response from the API.

        """
        
        if(translation_instructions is None):
            translation_instructions = OpenAIService._default_translation_instructions

        if(OpenAIService._decorator_to_use is None):
            return await OpenAIService.__translate_text_async(translation_instructions, translation_prompt)
        
        decorated_function = OpenAIService._decorator_to_use(OpenAIService.__translate_text_async)
        return await decorated_function(translation_instructions, translation_prompt)

##-------------------start-of-_translate_message()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __translate_text(instructions:SystemTranslationMessage, prompt:ModelTranslationMessage) -> ChatCompletion:

        """

        Synchronously translates the text using the OpenAI API.

        Parameters:
        instructions (SystemTranslationMessage) : The instructions to use for the translation.
        prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (ChatCompletion) : The response from the API.

        """

        response_format = "json_object" if OpenAIService._json_mode and OpenAIService._model in VALID_JSON_OPENAI_MODELS else "text"

        attributes = ["temperature", "logit_bias", "top_p", "n", "stream", "stop", "presence_penalty", "frequency_penalty", "max_tokens"]
        message_args = {
            "response_format": { "type": response_format },
            "model": OpenAIService._model,
            "messages": [instructions.to_dict(), prompt.to_dict()],
            ## scary looking dict comprehension to get the attributes that are not NOT_GIVEN
            **{attr: getattr(OpenAIService, f"_{attr}") for attr in attributes if getattr(OpenAIService, f"_{attr}") != NOT_GIVEN}
        }

        response = OpenAIService._sync_client.chat.completions.create(**message_args)
        
        return response
    
##-------------------start-of- __translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def __translate_text_async(instructions:SystemTranslationMessage, prompt:ModelTranslationMessage) -> ChatCompletion:

        """

        Asynchronously translates the text using the OpenAI API.

        Parameters:
        instruction (SystemTranslationMessage) : The instructions to use for the translation.
        prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (ChatCompletion) : The response from the API.

        """

        async with OpenAIService._semaphore:

            response_format = "json_object" if OpenAIService._json_mode and OpenAIService._model in VALID_JSON_OPENAI_MODELS else "text"

            if(OpenAIService._rate_limit_delay is not None):
                await asyncio.sleep(OpenAIService._rate_limit_delay)

            attributes = ["temperature", "logit_bias", "top_p", "n", "stream", "stop", "presence_penalty", "frequency_penalty", "max_tokens"]
            message_args = {
                "response_format": { "type": response_format },
                "model": OpenAIService._model,
                "messages": [instructions.to_dict(), prompt.to_dict()],
                ## scary looking dict comprehension to get the attributes that are not NOT_GIVEN
                **{attr: getattr(OpenAIService, f"_{attr}") for attr in attributes if getattr(OpenAIService, f"_{attr}") != NOT_GIVEN}
            }

            response = await OpenAIService._async_client.chat.completions.create(**message_args)
            
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

            OpenAIService._sync_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":"This is a test."}],
                max_tokens=1
            ) 

            _validity = True

            return _validity, None

        except Exception as _e:

            return _validity, _e

##-------------------start-of-_calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _calculate_cost(text:str | ModelTranslationMessage | SystemTranslationMessage | typing.Iterable, 
                        translation_instructions:str | None, 
                        model:str | None) -> typing.Tuple[int, float, str]:

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
            translation_instructions = OpenAIService._default_translation_instructions.content

        if(isinstance(text, typing.Iterable)):

            if(not isinstance(text,str) and not _is_iterable_of_strings(text)):
                raise ValueError("The text must be a string or an iterable of strings.")
            
            if(isinstance(text, ModelTranslationMessage) or isinstance(text, SystemTranslationMessage)):
                ## since instructions are paired with the text, we need to repeat the instructions for index
                ## this only works if the text is pre-built ModelTranslationMessage or SystemTranslationMessage objects
                ## otherwise, the instructions will be repeated for each item in the iterable which is not the intended behavior
                translation_instructions = translation_instructions * len(text) # type: ignore

            else:
                ## otherwise, we can really only estimate.
                cost_modifier = 2.5

            text = _convert_iterable_to_str(text)

        if(isinstance(translation_instructions, typing.Iterable)):
            translation_instructions = _convert_iterable_to_str(translation_instructions)

        if(model is None):
            model = OpenAIService._default_model

        ## not exactly how the text will be formatted, but it's close enough for the purposes of estimating the cost as tokens should be the same
        total_text_to_estimate = f"{translation_instructions}\n{text}"
        
        _num_tokens, _cost, _ = _estimate_cost(total_text_to_estimate, model)

        _cost = _cost * cost_modifier

        return _num_tokens, _cost, model   