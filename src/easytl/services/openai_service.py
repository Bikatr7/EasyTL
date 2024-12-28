## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio
from typing import AsyncIterator, Iterator

## third-party libraries
from pydantic import BaseModel
from openai import AsyncOpenAI, OpenAI

## custom modules
from ..classes import SystemTranslationMessage, ModelTranslationMessage, ChatCompletion, NOT_GIVEN, NotGiven
from ..exceptions import EasyTLException

from ..util.util import _convert_iterable_to_str, _estimate_cost, _is_iterable_of_strings
from ..util.constants import VALID_JSON_OPENAI_MODELS, VALID_STRUCTURED_OUTPUT_OPENAI_MODELS


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

    _json_mode:bool = False
    _response_schema:typing.Union[typing.Mapping[str, typing.Any], typing.Type[BaseModel], None] = None

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
                        semaphore:int | None=None,
                        rate_limit_delay:float | None=None,
                        json_mode:bool=False,
                        response_schema:typing.Union[typing.Mapping[str, typing.Any], typing.Type[BaseModel], None] = None
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

            OpenAIService._rate_limit_delay = rate_limit_delay

            OpenAIService._json_mode = json_mode
            OpenAIService._response_schema = response_schema

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

                if(OpenAIService._response_schema and OpenAIService._model in VALID_STRUCTURED_OUTPUT_OPENAI_MODELS):
                    OpenAIService._default_translation_instructions = SystemTranslationMessage("Please translate the following text into English. Make sure to return the translated text in JSON format according to the schema provided.")

                elif(OpenAIService._response_schema):
                    model_string = ", ".join(VALID_STRUCTURED_OUTPUT_OPENAI_MODELS)
                    raise EasyTLException("Structured output mode for OpenAI is only available for the following models: " + model_string)

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
    def _translate_text(translation_instructions: typing.Optional[SystemTranslationMessage],
                                translation_prompt: ModelTranslationMessage
                                ) -> ChatCompletion | Iterator[ChatCompletion]:
        
        """
        
        Synchronously translates the text using the OpenAI API.

        Parameters:
        translation_instructions (SystemTranslationMessage) : The instructions to use for the translation.
        translation_prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (ChatCompletion | Iterator[ChatCompletion]) : The response from the API. Returns an iterator if streaming is enabled.

        """
        
        if(translation_instructions is None):
            translation_instructions = OpenAIService._default_translation_instructions

        if(OpenAIService._decorator_to_use is None):
            return OpenAIService.__translate_text(translation_instructions, translation_prompt)

        decorated_function = OpenAIService._decorator_to_use(OpenAIService.__translate_text)
        return decorated_function(translation_instructions, translation_prompt)
    
##-------------------start-of-_translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def _translate_text_async(translation_instructions: typing.Optional[SystemTranslationMessage],
                                translation_prompt: ModelTranslationMessage
                                ) -> ChatCompletion | AsyncIterator[ChatCompletion]:
        
        """

        Asynchronously translates the text using the OpenAI API.

        Parameters:
        translation_instructions (SystemTranslationMessage) : The instructions to use for the translation.
        translation_prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (ChatCompletion | AsyncIterator[ChatCompletion]) : The response from the API. Returns an async iterator if streaming is enabled.

        """
        
        if(translation_instructions is None):
            translation_instructions = OpenAIService._default_translation_instructions

        if(OpenAIService._decorator_to_use is None):
            return await OpenAIService.__translate_text_async(translation_instructions, translation_prompt)
        
        decorated_function = OpenAIService._decorator_to_use(OpenAIService.__translate_text_async)
        return await decorated_function(translation_instructions, translation_prompt)

##-------------------start-of-_translate_message()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __translate_text(instructions:SystemTranslationMessage, prompt:ModelTranslationMessage) -> ChatCompletion | Iterator[ChatCompletion]:

        """

        Synchronously translates the text using the OpenAI API.

        Parameters:
        instructions (SystemTranslationMessage) : The instructions to use for the translation.
        prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (ChatCompletion | Iterator[ChatCompletion]) : The response from the API. Returns an iterator if streaming is enabled.

        """

        calling_function = OpenAIService._sync_client.chat.completions.create

        ## json mode; no schema
        if(OpenAIService._json_mode and OpenAIService._model in VALID_JSON_OPENAI_MODELS and not OpenAIService._response_schema):
            response_format = { "type": "json_object" }

        ## Json mode; manual schema
        elif (OpenAIService._json_mode and 
              OpenAIService._response_schema and 
              OpenAIService._model in VALID_STRUCTURED_OUTPUT_OPENAI_MODELS and 
              isinstance(OpenAIService._response_schema, type) and 
              not issubclass(OpenAIService._response_schema, BaseModel)):
                        
            response_format = { "type": "json_schema", "json_schema": OpenAIService._response_schema }
        
        ## Json mode; schema from BaseModel
        elif (OpenAIService._json_mode and 
              OpenAIService._response_schema and 
              OpenAIService._model in VALID_STRUCTURED_OUTPUT_OPENAI_MODELS and 
              isinstance(OpenAIService._response_schema, type) and 
              issubclass(OpenAIService._response_schema, BaseModel)):
            
            response_format = OpenAIService._response_schema

            calling_function = OpenAIService._sync_client.beta.chat.completions.parse

        ## Non-Json mode
        else:
            response_format = { "type": "text" }

        attributes = ["temperature", "logit_bias", "top_p", "n", "stream", "stop", "presence_penalty", "frequency_penalty", "max_tokens"]
        message_args = {
            "response_format": response_format,
            "model": OpenAIService._model,
            "messages": [instructions.to_dict(), prompt.to_dict()],
            "stream": OpenAIService._stream,
            **{attr: getattr(OpenAIService, f"_{attr}") for attr in attributes if getattr(OpenAIService, f"_{attr}") != NOT_GIVEN and attr != "stream"}
        }

        ## remove max_tokens from the message args if using o1 as it's not supported
        if(OpenAIService._model in ["o1-2024-09-12", "o1"]):
            message_args.pop("max_tokens", None)

        ## remove stream from the message args since it's not needed for the parse function (or at all really)
        if(calling_function == OpenAIService._sync_client.beta.chat.completions.parse):
            message_args.pop("stream", None)

        response = calling_function(**message_args)
        return response
    
##-------------------start-of- __translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def __translate_text_async(instructions:SystemTranslationMessage, prompt:ModelTranslationMessage) -> ChatCompletion | AsyncIterator[ChatCompletion]:

        """

        Asynchronously translates the text using the OpenAI API.

        Parameters:
        instruction (SystemTranslationMessage) : The instructions to use for the translation.
        prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (ChatCompletion | AsyncIterator[ChatCompletion]) : The response from the API. Returns an async iterator if streaming is enabled.

        """

        calling_function = OpenAIService._async_client.chat.completions.create

        ## json mode; no schema
        if(OpenAIService._json_mode and OpenAIService._model in VALID_JSON_OPENAI_MODELS and not OpenAIService._response_schema):
            response_format = { "type": "json_object" }
        
        ## Json mode; manual schema
        elif(OpenAIService._json_mode and 
              OpenAIService._response_schema and 
              OpenAIService._model in VALID_STRUCTURED_OUTPUT_OPENAI_MODELS and 
              not isinstance(OpenAIService._response_schema, type)):

            response_format = { "type": "json_schema", "json_schema": OpenAIService._response_schema }
        
        ## Json mode; schema from BaseModel
        elif(OpenAIService._json_mode and 
              OpenAIService._response_schema and 
              OpenAIService._model in VALID_STRUCTURED_OUTPUT_OPENAI_MODELS and 
              isinstance(OpenAIService._response_schema, type) and 
              issubclass(OpenAIService._response_schema, BaseModel)):

            response_format = OpenAIService._response_schema

            calling_function = OpenAIService._async_client.beta.chat.completions.parse

        ## Non-Json mode
        else:
            response_format = { "type": "text" }


        if(OpenAIService._rate_limit_delay is not None):
            await asyncio.sleep(OpenAIService._rate_limit_delay)

        attributes = ["temperature", "logit_bias", "top_p", "n", "stream", "stop", "presence_penalty", "frequency_penalty", "max_tokens"]
        message_args = {
            "response_format": response_format,
            "model": OpenAIService._model,
            "messages": [instructions.to_dict(), prompt.to_dict()],
            "stream": OpenAIService._stream,
            **{attr: getattr(OpenAIService, f"_{attr}") for attr in attributes if getattr(OpenAIService, f"_{attr}") != NOT_GIVEN and attr != "stream"}
        }

        ## remove max_tokens from the message args if using o1 as it's not supported
        if(OpenAIService._model in ["o1-2024-09-12", "o1"]):
            message_args.pop("max_tokens", None)

        ## remove stream from the message args since it's not needed for the parse function (or at all really)
        if(calling_function == OpenAIService._async_client.beta.chat.completions.parse):
            message_args.pop("stream")

        response = await calling_function(**message_args)
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
                model="gpt-4o",
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