## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing

## third-party libraries
from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion

## custom modules
from .classes import SystemTranslationMessage, ModelTranslationMessage
from .util import _convert_iterable_to_str, _estimate_cost, _is_iterable_of_strings

class OpenAIService:

    _default_model:str = "gpt-4"
    _default_translation_instructions:SystemTranslationMessage = SystemTranslationMessage("Please translate the following text into English.")

    _system_message:typing.Optional[typing.Union[SystemTranslationMessage, str]] = _default_translation_instructions

    _model:str = _default_model
    _temperature:float = 0.3
    _logit_bias:typing.Dict[str, int] | None
    _top_p:float = 1.0
    _n:int = 1
    _stream:bool = False
    _stop:typing.List[str] | None = None
    _max_tokens:int | None = None
    _presence_penalty:float = 0.0
    _frequency_penalty:float = 0.0


    _sync_client = OpenAI(max_retries=0, api_key="DummyKey")
    _async_client = AsyncOpenAI(max_retries=0, api_key="DummyKey")

    _decorator_to_use:typing.Union[typing.Callable, None] = None

##-------------------start-of-set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_api_key(api_key:str) -> None:

        """

        Sets the API key for the OpenAI client.

        Parameters:
        api_key (string) : The API key to set.

        """

        OpenAIService._async_client.api_key = api_key
        OpenAIService._sync_client.api_key = api_key

##-------------------start-of-set_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_decorator(decorator:typing.Callable) -> None:

        """

        Sets the decorator to use for the OpenAI service. Should be a callable that returns a decorator.

        Parameters:
        decorator (callable) : The decorator to use.

        """

        OpenAIService._decorator_to_use = decorator

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def _set_attributes(model:str = _default_model,
                        temperature:float = 0.3,
                        logit_bias:typing.Dict[str, int] | None = None,
                        top_p:float = 1.0,
                        n:int = 1,
                        stream:bool = False,
                        stop:typing.List[str] | None = None,
                        max_tokens:int | None = None,
                        presence_penalty:float = 0.0,
                        frequency_penalty:float = 0.0) -> None:
    
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

        response = OpenAIService._sync_client.chat.completions.create(
            messages=[
                instructions.to_dict(),
                prompt.to_dict()
            ], # type: ignore

            model=OpenAIService._model,
            temperature=OpenAIService._temperature,
            logit_bias=OpenAIService._logit_bias,
            top_p=OpenAIService._top_p,
            n=OpenAIService._n,
            stream=OpenAIService._stream,
            stop=OpenAIService._stop,
            presence_penalty=OpenAIService._presence_penalty,
            frequency_penalty=OpenAIService._frequency_penalty,
            max_tokens=OpenAIService._max_tokens
            
        ) 
        
        return response
    
##-------------------start-of- __translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def __translate_text_async(instruction:SystemTranslationMessage, prompt:ModelTranslationMessage) -> ChatCompletion:

        """

        Asynchronously translates the text using the OpenAI API.

        Parameters:
        instruction (SystemTranslationMessage) : The instructions to use for the translation.
        prompt (ModelTranslationMessage) : The text to translate.

        Returns:
        response (ChatCompletion) : The response from the API.

        """

        response = await OpenAIService._async_client.chat.completions.create(
            messages=[
                instruction.to_dict(),
                prompt.to_dict()
            ],  # type: ignore

            model=OpenAIService._model,
            temperature=OpenAIService._temperature,
            logit_bias=OpenAIService._logit_bias,
            top_p=OpenAIService._top_p,
            n=OpenAIService._n,
            stream=OpenAIService._stream,
            stop=OpenAIService._stop,
            presence_penalty=OpenAIService._presence_penalty,
            frequency_penalty=OpenAIService._frequency_penalty,
            max_tokens=OpenAIService._max_tokens
            
        ) 

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
        
##-------------------start-of-get_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _get_decorator() -> typing.Union[typing.Callable, None]:

        """

        Returns the decorator to use for the OpenAI service.

        Returns:
        decorator (callable) : The decorator to use.

        """

        return OpenAIService._decorator_to_use
    
##-------------------start-of-_calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _calculate_cost(text:str | typing.Iterable, translation_instructions:str | None, model:str | None) -> typing.Tuple[int, float, str]:

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

        if(translation_instructions is None):
            translation_instructions = OpenAIService._default_translation_instructions.content

        if(isinstance(text, typing.Iterable)):

            if(not isinstance(text,str) and not _is_iterable_of_strings(text)):
                raise ValueError("The text must be a string or an iterable of strings.")

            ## since instructions are paired with the text, we need to repeat the instructions for index
            translation_instructions = translation_instructions * len(text) # type: ignore

            text = _convert_iterable_to_str(text)

        if(isinstance(translation_instructions, typing.Iterable)):
            translation_instructions = _convert_iterable_to_str(translation_instructions)

        if(model is None):
            model = OpenAIService._default_model

        ## not exactly how the text will be formatted, but it's close enough for the purposes of estimating the cost as tokens should be the same
        total_text_to_estimate = f"{translation_instructions}\n{text}"
        
        _num_tokens, _cost, _ = _estimate_cost(total_text_to_estimate, model)

        return _num_tokens, _cost, model   