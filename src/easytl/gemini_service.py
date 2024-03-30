## built-in libraries
import typing

## third party libraries
from google.generativeai import GenerationConfig
import google.generativeai as genai

class GeminiService:

    _default_translation_instructions:str = "Please translate the following text into English."

    _model:str = "gemini-pro"
    _text_to_translate:str = ""
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

    ## I don't plan to allow users to change these settings, as I believe that translations should be as accurate as possible, avoiding any censorship or filtering of content.
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

##-------------------start-of-_set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_api_key(api_key:str) -> None:

        """

        Sets the API key for the Gemini client.

        Parameters:
        api_key (string) : The API key.

        """

        genai.configure(api_key=api_key)

##-------------------start-of-_set_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_decorator(decorator:typing.Callable) -> None:

        """

        Sets the decorator to use for the Gemini service. Should be a callable that returns a decorator.

        Parameters:
        decorator (callable) : The decorator to use.

        """

        GeminiService._decorator_to_use = decorator

##-------------------start-of-_set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def _set_attributes(model:str="gemini-pro",
                        text_to_translate:str="",
                        temperature:float=0.5,
                        top_p:float=0.9,
                        top_k:int=40,
                        candidate_count:int=1,
                        stream:bool=False,
                        stop_sequences:typing.List[str] | None=None,
                        max_output_tokens:int | None=None) -> None:
        
        """

        Sets the attributes for the Gemini service.

        """

        GeminiService._model = model
        GeminiService._text_to_translate = text_to_translate
        GeminiService._temperature = temperature
        GeminiService._top_p = top_p
        GeminiService._top_k = top_k
        GeminiService._candidate_count = candidate_count
        GeminiService._stream = stream
        GeminiService._stop_sequences = stop_sequences
        GeminiService._max_output_tokens = max_output_tokens
        
##-------------------start-of-_redefine_client()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _redefine_client() -> None:

        """

        Redefines the Gemini client and generation config. This should be called before making any requests to the Gemini service, or after changing any of the service's settings.

        """

        GeminiService._client = genai.GenerativeModel(model_name=GeminiService._model,
                                                     safety_settings=GeminiService._safety_settings)

        GeminiService._generation_config = GenerationConfig(candidate_count=GeminiService._candidate_count,
                                                           max_output_tokens=GeminiService._max_output_tokens,
                                                           stop_sequences=GeminiService._stop_sequences,
                                                            temperature=GeminiService._temperature,
                                                            top_p=GeminiService._top_p,
                                                            top_k=GeminiService._top_k)

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
            
##-------------------start-of-count_tokens()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def count_tokens(text:str) -> int:

        """

        Counts the number of tokens in the given text.

        Parameters:
        text (string) : The text to count tokens in.

        Returns:
        total_tokens (int) : The number of tokens in the text.

        """

        return genai.GenerativeModel(GeminiService._model).count_tokens(text).total_tokens

##-------------------start-of-_translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    @_redefine_client_decorator
    async def _translate_text_async(text_to_translate:str, translation_instructions:typing.Optional[str]) -> str:

        """

        Asynchronously translates a text.
        Instructions default to translating whatever text is input into English.

        Parameters:
        text_to_translate (string) : The text to translate.
        translation_instructions (string) : The instructions for the translation.

        Returns:
        output (string) : The translation.

        """

        if(translation_instructions is None):
            translation_instructions = GeminiService._default_translation_instructions

        if(GeminiService._decorator_to_use is None):
            return await GeminiService.__translate_text_async(translation_instructions, text_to_translate)

        decorated_function = GeminiService._decorator_to_use(GeminiService.__translate_text_async)
        return await decorated_function(translation_instructions, text_to_translate)
    
##-------------------start-of-_translate_message()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    @_redefine_client_decorator
    def _translate_message(text_to_translate:str, translation_instructions:typing.Optional[str]) -> str:

        """

        Synchronously translates a text.
        Instructions default to translating whatever text is input into English.

        Parameters:
        text_to_translate (string) : The text to translate.
        translation_instructions (string) : The instructions for the translation.

        Returns:
        output (string) : The translation.

        """

        if(translation_instructions is None):
            translation_instructions = GeminiService._default_translation_instructions

        if(GeminiService._decorator_to_use is None):
            return GeminiService.__translate_text(translation_instructions, text_to_translate)

        decorated_function = GeminiService._decorator_to_use(GeminiService.__translate_text)
        return decorated_function(translation_instructions, text_to_translate)
    
##-------------------start-of-__translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def __translate_text(translation_instructions:str, text_to_translate:str) -> str:

        """

        Synchronously translates text.

        Parameters:
        translation_instructions (string) : The instructions for the translation.
        text_to_translate (string) : The text to translate.

        Returns:
        output (string) : The translation.

        """

        _response = GeminiService._client.generate_content(
            contents=translation_instructions + "\n" + text_to_translate,
            generation_config=GeminiService._generation_config,
            safety_settings=GeminiService._safety_settings,
            stream=GeminiService._stream
        )
        
        ## may need to add some error handling here later for if the response is not successful

        return _response.text

##-------------------start-of-__translate_message_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def __translate_text_async(translation_instructions:str, text_to_translate:str) -> str:

        """

        Asynchronously translates text.

        Parameters:
        translation_instructions (string) : The instructions for the translation.
        text_to_translate (string) : The text to translate.

        Returns:
        output (string) : The translation.

        """

        _response = await GeminiService._client.generate_content_async(
            contents=translation_instructions + "\n" + text_to_translate,
            generation_config=GeminiService._generation_config,
            safety_settings=GeminiService._safety_settings,
            stream=GeminiService._stream
        )
        
        ## may need to add some error handling here later for if the response is not successful

        return _response.text
    
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
        
##-------------------start-of-_get_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _get_decorator() -> typing.Union[typing.Callable, None]:

        """

        Returns the decorator to use for the Gemini service.

        Returns:
        decorator (callable) : The decorator to use.

        """

        return GeminiService._decorator_to_use