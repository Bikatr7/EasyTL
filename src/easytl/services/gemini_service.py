## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio

import google.generativeai as genai

## custom modules
from ..util.util import _estimate_cost, _convert_iterable_to_str, _is_iterable_of_strings
from ..util.constants import VALID_JSON_GEMINI_MODELS as VALID_SYSTEM_MESSAGE_MODELS, VALID_JSON_GEMINI_MODELS

from ..classes import GenerationConfig, GenerateContentResponse, AsyncGenerateContentResponse
from ..exceptions import EasyTLException, InvalidTextInputException

class GeminiService:

    _default_translation_instructions:str = "Please translate the following text into English."
    _default_model:str = "gemini-pro"

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

    _semaphore_value:int = 5
    _semaphore:asyncio.Semaphore = asyncio.Semaphore(_semaphore_value)

    _rate_limit_delay:float | None = None

    _decorator_to_use:typing.Union[typing.Callable, None] = None

    ## Set to prevent any blockage of content
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

    _json_mode:bool = False
    _response_schema:typing.Mapping[str, typing.Any] | None = None

##-------------------start-of-_set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_api_key(api_key:str) -> None:

        """

        Sets the API key for the Gemini client.

        Parameters:
        api_key (string) : The API key.

        """

        genai.configure(api_key=api_key)

##-------------------start-of-_set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def _set_attributes(model:str="gemini-pro",
                        system_message:str | None = _default_translation_instructions,
                        temperature:float=0.5,
                        top_p:float=0.9,
                        top_k:int=40,
                        candidate_count:int=1,
                        stream:bool=False,
                        stop_sequences:typing.List[str] | None=None,
                        max_output_tokens:int | None=None,
                        decorator:typing.Union[typing.Callable, None]=None,
                        semaphore:int | None=None,
                        rate_limit_delay:float | None=None,
                        json_mode:bool=False,
                        response_schema:typing.Mapping[str, typing.Any] | None = None
                        ) -> None:
        
        """

        Sets the attributes for the Gemini service.

        """

        GeminiService._model = model
        GeminiService._system_message = system_message
        GeminiService._temperature = temperature
        GeminiService._top_p = top_p
        GeminiService._top_k = top_k
        GeminiService._candidate_count = candidate_count
        GeminiService._stream = stream
        GeminiService._stop_sequences = stop_sequences
        GeminiService._max_output_tokens = max_output_tokens

        GeminiService._decorator_to_use = decorator

        GeminiService._rate_limit_delay = rate_limit_delay

        GeminiService._json_mode = json_mode
        GeminiService._response_schema = response_schema

        # if a semaphore is not provided, set it to the default value based on the model
        ## rate limits for 1.5 models are 2 requests per second
        ## rate limits for 1.0B models are 5 requests per second
        semaphore_values = {"gemini-1.5-pro": 2,
                            "gemini-1.5-pro-001": 2,
                            "gemini-1.5-pro-002": 2,
                            "gemini-1.5-flash": 2,
                            "gemini-1.5-flash-001": 2,
                            "gemini-1.5-flash-002": 2,
                            "gemini-1.5-pro-latest": 2,
                            "gemini-1.5-flash-latest": 2,
                            }
        
        GeminiService._semaphore_value = semaphore or semaphore_values.get(GeminiService._model, 5)
        GeminiService._semaphore = asyncio.Semaphore(GeminiService._semaphore_value)

        if(GeminiService._json_mode and GeminiService._model in VALID_JSON_GEMINI_MODELS and response_schema is not None):
            GeminiService._default_translation_instructions = "Please translate the following text into English. Make sure to return the translated text in JSON format. The JSON should be in the format specified in the response schema."

        elif(GeminiService._json_mode and GeminiService._model in VALID_JSON_GEMINI_MODELS):
            GeminiService._default_translation_instructions = "Please translate the following text into English. Make sure to return the translated text in JSON format."

        elif(GeminiService._json_mode):
            allowed_models_string = ", ".join(VALID_JSON_GEMINI_MODELS)
            raise EasyTLException(f"JSON mode for Gemini is only supported for the following models: {allowed_models_string}")
        
        else:
            GeminiService._default_translation_instructions = "Please translate the following text into English."
        
##-------------------start-of-_redefine_client()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _redefine_client() -> None:

        """

        Redefines the Gemini client and generation config. This should be called before making any requests to the Gemini service, or after changing any of the service's settings.

        """

        response_mime_type = "application/json" if GeminiService._json_mode else "text/plain"
        
        gen_model_params = {
            "model_name": GeminiService._model,
            "safety_settings": GeminiService._safety_settings,
            "system_instruction": GeminiService._system_message if GeminiService._model in VALID_SYSTEM_MESSAGE_MODELS else None
        }
        
        GeminiService._client = genai.GenerativeModel(**gen_model_params)
        
        generation_config_params = {
            "candidate_count": GeminiService._candidate_count,
            "stop_sequences": GeminiService._stop_sequences,
            "max_output_tokens": GeminiService._max_output_tokens,
            "temperature": GeminiService._temperature,
            "top_p": GeminiService._top_p,
            "top_k": GeminiService._top_k,
            "response_mime_type": response_mime_type,
            "response_schema": GeminiService._response_schema if GeminiService._response_schema and GeminiService._json_mode else None
        }
        
        GeminiService._generation_config = GenerationConfig(**generation_config_params)
        
        GeminiService._semaphore = asyncio.Semaphore(GeminiService._semaphore_value)

##-------------------start-of-_redefine_client_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _redefine_client_decorator(func):

        """

        Wraps a function to redefine the Gemini client before doing anything that requires the client.

        Parameters:
        func (callable) : The function to wrap.

        Returns:
        wrapper (callable) : The wrapped function.

        """

        def wrapper(*args, **kwargs):
            GeminiService._redefine_client() 
            return func(*args, **kwargs)
        
        return wrapper
            
##-------------------start-of-_translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    @_redefine_client_decorator
    async def _translate_text_async(text_to_translate:str) -> AsyncGenerateContentResponse:

        """

        Asynchronously translates text.
        Instructions default to translating whatever text is input into English.

        Parameters:
        text_to_translate (string) : The text to translate.

        Returns:
        AsyncGenerateContentResponse : The translation.

        """

        if(GeminiService._decorator_to_use is None):
            return await GeminiService.__translate_text_async(text_to_translate)

        _decorated_function = GeminiService._decorator_to_use(GeminiService.__translate_text_async)
        return await _decorated_function(text_to_translate)
    
##-------------------start-of-_translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    @_redefine_client_decorator
    def _translate_text(text_to_translate:str) -> GenerateContentResponse | typing.Iterator[GenerateContentResponse]:

        """

        Synchronously translates text.
        Instructions default to translating whatever text is input into English.

        Parameters:
        text_to_translate (string) : The text to translate.

        Returns:
        GenerateContentResponse | Iterator[GenerateContentResponse] : The translation. Returns an iterator if streaming is enabled.

        """

        if(GeminiService._decorator_to_use is None):
            return GeminiService.__translate_text(text_to_translate)

        _decorated_function = GeminiService._decorator_to_use(GeminiService.__translate_text)
        return _decorated_function(text_to_translate)
    
##-------------------start-of-__translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def __translate_text(text_to_translate:str) -> GenerateContentResponse | typing.Iterator[GenerateContentResponse]:

        """

        Synchronously translates text.

        Parameters:
        text_to_translate (string) : The text to translate.

        Returns:
        GenerateContentResponse | Iterator[GenerateContentResponse] : The translation. Returns an iterator if streaming is enabled.

        """

        text_request = f"{text_to_translate}" if GeminiService._model in VALID_SYSTEM_MESSAGE_MODELS else f"{GeminiService._system_message}\n{text_to_translate}"

        _response = GeminiService._client.generate_content(
            contents=text_request,
            generation_config=GeminiService._generation_config,
            safety_settings=GeminiService._safety_settings,
            stream=GeminiService._stream
        )
        
        return _response

##-------------------start-of-__translate_message_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def __translate_text_async(text_to_translate:str) -> AsyncGenerateContentResponse:

        """

        Asynchronously translates text.

        Parameters:
        text_to_translate (string) : The text to translate.

        Returns:
        _response (AsyncGenerateContentResponse) : The translation.

        """

        async with GeminiService._semaphore:

            if(GeminiService._rate_limit_delay is not None):
                await asyncio.sleep(GeminiService._rate_limit_delay)

            text_request = f"{text_to_translate}" if GeminiService._model in VALID_SYSTEM_MESSAGE_MODELS else f"{GeminiService._system_message}\n{text_to_translate}"

            _response = await GeminiService._client.generate_content_async(
                contents=text_request,
                generation_config=GeminiService._generation_config,
                safety_settings=GeminiService._safety_settings,
                stream=GeminiService._stream
            )
            
            return _response
        
##-------------------start-of-test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    @_redefine_client_decorator
    def _test_api_key_validity() -> typing.Tuple[bool, typing.Union[Exception, None]]:

        """

        Tests the validity of the API key.

        Returns:
        validity (bool) : True if the API key is valid, False if it is not.
        e (Exception) : The exception that was raised, if any.

        """

        _validity = False

        try:

            _generation_config = GenerationConfig(candidate_count=1, max_output_tokens=1)

            GeminiService._client.generate_content(
                "Respond to this with 1",generation_config=_generation_config
            )

            _validity = True

            return _validity, None

        except Exception as _e:

            return _validity, _e
        
##-------------------start-of-_calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    @_redefine_client_decorator
    def _calculate_cost(text:str | typing.Iterable, translation_instructions:str | None, model:str | None) -> typing.Tuple[int, float, str]:

        """

        Calculates the cost of the translation.

        Parameters:
        text (string | iterable) : The text to translate.
        translation_instructions (string) : The instructions for the translation.
        model (string) : The model to use for the translation.

        Returns:
        num_tokens (int) : The number of tokens in the text.
        cost (float) : The cost of the translation.
        model (string) : The model used for the translation.

        """

        if(translation_instructions is None):
            translation_instructions = GeminiService._default_translation_instructions

        if(isinstance(text, typing.Iterable)):

            if(not isinstance(text,str) and not _is_iterable_of_strings(text)):
                raise InvalidTextInputException("The text must be a string or an iterable of strings.")

            ## since instructions are paired with the text, we need to repeat the instructions for index
            translation_instructions = translation_instructions * len(text) # type: ignore

            text = _convert_iterable_to_str(text)

        if(isinstance(translation_instructions, typing.Iterable)):
            translation_instructions = _convert_iterable_to_str(translation_instructions)

        if(model is None):
            model = GeminiService._default_model

        ## not exactly how the text will be formatted, but it's close enough for the purposes of estimating the cost as tokens should be the same
        total_text_to_estimate = f"{translation_instructions}\n{text}"
        
        _num_tokens, _cost, model = _estimate_cost(total_text_to_estimate, model)

        return _num_tokens, _cost, model  