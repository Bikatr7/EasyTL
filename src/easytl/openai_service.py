## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing

## third-party libraries
from openai import AsyncOpenAI, OpenAI

## custom modules
from .exceptions import InvalidAPIKeyException
from .classes import SystemTranslationMessage, Message

class OpenAIService:

    _default_model:str = "gpt-4"
    _default_translation_instructions:typing.Union[SystemTranslationMessage, str] = SystemTranslationMessage("Please translate the following text into English.")

    _system_message:typing.Optional[typing.Union[SystemTranslationMessage, str]] = _default_translation_instructions

    _model:str = _default_model
    _system_message:typing.Optional[typing.Union[SystemTranslationMessage, str]] = _default_translation_instructions
    _temperature:float = 0.3
    _logit_bias:typing.Dict[str, float] | None
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
    def set_api_key(api_key:str) -> None:

        """

        Sets the API key for the OpenAI _async_client.

        Parameters:
        api_key (string) : The API key to set.

        """

        OpenAIService._async_client.api_key = api_key

##-------------------start-of-set_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def set_decorator(decorator:typing.Callable) -> None:

        """

        Sets the decorator to use for the OpenAI service. Should be a callable that returns a decorator.

        Parameters:
        decorator (callable) : The decorator to use.

        """

        OpenAIService.decorator_to_use = decorator

##-------------------start-of-trans()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def translate_message(translation_instructions:Message, translation_prompt:Message) -> str:

        """
        
        Translates a system and user message.

        Parameters:
        translation_instructions (object - SystemTranslationMessage | ModelTranslationMessage) : The system message also known as the instructions.
        translation_prompt (object - ModelTranslationMessage) : The user message also known as the prompt.

        Returns:
        output (string) a string that gpt gives to us also known as the translation.

        """

        if(OpenAIService.decorator_to_use == None):
            return await OpenAIService._translate_message(translation_instructions, translation_prompt)

        decorated_function = OpenAIService.decorator_to_use(OpenAIService._translate_message)
        return await decorated_function(translation_instructions, translation_prompt)

##-------------------start-of-_translate_message()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ## backoff wrapper for retrying on errors, As of OpenAI > 1.0.0, it comes with a built in backoff system, but I've grown accustomed to this one so I'm keeping it.
    @staticmethod
    async def _translate_message(translation_instructions:Message, translation_prompt:Message) -> str:

        """

        Translates a system and user message.

        Parameters:
        translation_instructions (object - SystemTranslationMessage | ModelTranslationMessage) : The system message also known as the instructions.
        translation_prompt (object - ModelTranslationMessage) : The user message also known as the prompt.

        Returns:
        output (string) a string that gpt gives to us also known as the translation.

        """

        if(OpenAIService._async_client.api_key == "DummyKey"):
            raise InvalidAPIKeyException("OpenAI")

        ## logit bias is currently excluded due to a lack of need, and the fact that i am lazy

        response = await OpenAIService._async_client.chat.completions.create(
            _model=OpenAIService._model,
            messages=[
                translation_instructions.to_dict(),
                translation_prompt.to_dict()
            ],  # type: ignore

            temperature = OpenAIService._temperature,
            top_p = OpenAIService._top_p,
            n = OpenAIService._n,
            stream = OpenAIService._stream,
            stop = OpenAIService._stop,
            presence_penalty = OpenAIService._presence_penalty,
            frequency_penalty = OpenAIService._frequency_penalty,
            max_tokens = OpenAIService._max_tokens       

        )

        ## if anyone knows how to type hint this please let me know
        output = response.choices[0].message.content
        
        return output
    
##-------------------start-of-test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    async def test_api_key_validity() -> typing.Tuple[bool, typing.Union[Exception, None]]:

        """

        Tests the validity of the API key.

        Returns:
        validity (bool) : True if the API key is valid, False if it is not.
        e (Exception) : The exception that was raised, if any.

        """

        validity = False

        try:

            await OpenAIService._async_client.chat.completions.create(
                _model="gpt-3.5-turbo",
                messages=[{"role":"user","content":"This is a test."}],
                max_tokens=1
            ) ## type: ignore we'll deal with this later

            validity = True

            return validity, None

        except Exception as e:

            return validity, e
        
##-------------------start-of-get_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_decorator() -> typing.Union[typing.Callable, None]:

        """

        Returns the decorator to use for the OpenAI service.

        Returns:
        decorator (callable) : The decorator to use.

        """

        return OpenAIService.decorator_to_use