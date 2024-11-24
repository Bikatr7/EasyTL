## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio
import os

## third-party libraries
from pydantic import BaseModel

## custom modules
from .services.deepl_service import DeepLService
from .services.gemini_service import GeminiService
from .services.openai_service import OpenAIService
from .services.googletl_service import GoogleTLService
from .services.anthropic_service import AnthropicService
from .services.azure_service import AzureService

from .classes import Language, SplitSentences, Formality, GlossaryInfo, NOT_GIVEN, NotGiven
from .classes import ModelTranslationMessage, SystemTranslationMessage, TextResult, GenerateContentResponse, AsyncGenerateContentResponse, ChatCompletion, AnthropicMessage, AnthropicTextBlock, AnthropicToolUseBlock
from .exceptions import InvalidAPITypeException, InvalidResponseFormatException, InvalidTextInputException, EasyTLException, InvalidAPIKeyException

from .util.util import _is_iterable_of_strings
from .util.llm_util import _validate_easytl_llm_translation_settings, _return_curated_gemini_settings, _return_curated_openai_settings, _validate_stop_sequences, _validate_response_schema,  _return_curated_anthropic_settings, _validate_text_length 

class EasyTL:

    """
    
    EasyTL global client, used to interact with Translation APIs.

    Use set_credentials() to set the credentials for the specified API type. (e.g. set_credentials("deepl", "your_api_key") or set_credentials("google translate", "path/to/your/credentials.json"))

    Use test_credentials() to test the validity of the credentials for the specified API type. (e.g. test_credentials("deepl")) (Optional) Done automatically when translating.

    Use translate() to translate text using the specified service with it's appropriate kwargs. Or specify the service by calling the specific translation function. (e.g. openai_translate())

    Use calculate_cost() to calculate the cost of translating text using the specified service. (Optional)

    See the documentation for each function for more information.

    """

##-------------------start-of-set_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def set_credentials(api_type:typing.Literal["deepl", "gemini", "openai", "google translate", "anthropic", "azure"], credentials:typing.Union[str, None] = None) -> None:

        """

        Sets the credentials for the specified API type.

        Parameters:
        api_type (literal["deepl", "gemini", "openai", "google translate", "anthropic", "azure"]) : The API type to set the credentials for.
        credentials (string) : The credentials to set. This is an api key for deepl, gemini, anthropic, azure and openai. For google translate, this is a path to your json that has your service account key.

        """

        service_map = {
            "deepl": DeepLService._set_api_key,
            "gemini": GeminiService._set_api_key,
            "openai": OpenAIService._set_api_key,
            "google translate": GoogleTLService._set_credentials,
            "anthropic": AnthropicService._set_api_key,
            "azure": AzureService._set_api_key

        }

        environment_map = {
            "deepl": "DEEPL_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google translate": "PATH_TO_GOOGLE_CREDENTIALS_JSON",
            "anthropic": "ANTHROPIC_API_KEY",
        }

        assert api_type in service_map, InvalidAPITypeException("Invalid API type specified. Supported types are 'deepl', 'gemini', 'openai', 'google translate', 'anthropic' and 'azure'.")

        # If credentials are not passed, check the environment variables
        if(credentials is None and os.environ.get(environment_map[api_type]) is not None):
            credentials = os.environ.get(environment_map[api_type])

        # If credentials are still None, raise an exception
        assert credentials is not None, InvalidAPIKeyException(f"No credentials provided for {api_type}. Please provide the credentials or set the environment variable {environment_map[api_type]} with the credentials.")

        service_map[api_type](credentials)

##-------------------start-of-test_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def test_credentials(api_type:typing.Literal["deepl", "gemini", "openai", "google translate", "anthropic", "azure"], azure_region:str = "westus") -> typing.Tuple[bool, typing.Optional[Exception]]:

        """

        Tests the validity of the credentials for the specified API type.

        Parameters:
        api_type (literal["deepl", "gemini", "openai", "google translate", "anthropic", "azure"]) : The API type to test the credentials for.
        azure_region (string) : The Azure region to test the credentials for. Default is 'westus'. Can be omitted for other services.

        Returns:
        (bool) : Whether the credentials are valid.
        (Exception) : The exception that was raised, if any. None otherwise.

        """

        api_services = {
            "deepl": {"service": DeepLService, "test_func": DeepLService._test_api_key_validity},
            "gemini": {"service": GeminiService, "test_func": GeminiService._test_api_key_validity},
            "openai": {"service": OpenAIService, "test_func": OpenAIService._test_api_key_validity},
            "google translate": {"service": GoogleTLService, "test_func": GoogleTLService._test_credentials},
            "anthropic": {"service": AnthropicService, "test_func": AnthropicService._test_api_key_validity},
            "azure": {"service": AzureService, "test_func": AzureService._test_credentials}
        }

        assert api_type in api_services, InvalidAPITypeException("Invalid API type specified. Supported types are 'deepl', 'gemini', 'openai', 'google translate', 'anthropic' and 'azure'.")

        if(api_type == "azure"):
            _region = azure_region

            if(_region is None and os.environ.get("AZURE_REGION") is not None):
                _region = os.environ.get("AZURE_REGION")
                print(f"Using Azure region from environment variable: {_region}")
                
            _, _e = api_services[api_type]["test_func"](_region)
        
        else:
            _, _e = api_services[api_type]["test_func"]()

        if(_e is not None):
            raise _e

        return True, None
    
##-------------------start-of-googletl_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def googletl_translate(text:typing.Union[str, typing.Iterable[str]],
                           target_lang:str = "en",
                           override_previous_settings:bool = True,
                           decorator:typing.Callable | None = None,
                           logging_directory:str | None = None,
                           response_type:typing.Literal["text", "raw"] | None = "text",
                           translation_delay:float | None = None,
                           format:typing.Literal["text", "html"] = "text",
                           source_lang:str | None = None) -> typing.Union[typing.List[str], str, typing.List[typing.Any], typing.Any]:
        
        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for specific translation functions and will be ignored")
            logging_directory = None

        """

        Translates the given text to the target language using Google Translate.

        This function assumes that the credentials have already been set.

        It is unknown whether Google Translate has backoff retrying implemented. Assume it does not exist.

        Google Translate v2 API is poorly documented and type hints are near non-existent. typing.Any return types are used for the raw response type.

        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Google Translate function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        response_type (literal["text", "raw"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        format (string or None) : The format of the text. Can be 'text' or 'html'. Default is 'text'. Google Translate appears to be able to translate html but this has not been tested thoroughly by EasyTL.
        source_lang (string or None) : The source language to translate from.

        Returns:
        result (string or list - string or any or list - any) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of any objects if the response type is 'raw' and input was an iterable, an any object otherwise.

        """

        assert response_type in ["text", "raw"], InvalidResponseFormatException("Invalid response type specified. Must be 'text' or 'raw'.")

        assert format in ["text", "html"], InvalidResponseFormatException("Invalid format specified. Must be 'text' or 'html'.")

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("google translate")

        if(override_previous_settings == True):
            GoogleTLService._set_attributes(target_language=target_lang, 
                                            format=format, 
                                            source_language=source_lang, 
                                            decorator=decorator, 
                                            semaphore=None, 
                                            rate_limit_delay=translation_delay)
            
        ## This section may seem overly complex, but it is necessary to apply the decorator outside of the function call to avoid infinite recursion.
        ## Attempting to dynamically apply the decorator within the function leads to unexpected behavior, where this function's arguments are passed to the function instead of the intended translation function.

        def translate(text):
            return GoogleTLService._translate_text(text)
        
        if(decorator is not None):
            translate = GoogleTLService._decorator_to_use(GoogleTLService._translate_text) ## type: ignore

        else:
            translate = GoogleTLService._translate_text
            
        if(isinstance(text, str)):
            result = translate(text)
        
            assert not isinstance(result, list), EasyTLException("Malformed response received. Please try again.")

            result = result if response_type == "raw" else result["translatedText"]
        
        elif(_is_iterable_of_strings(text)):

            results = [translate(t) for t in text]

            assert isinstance(results, list), EasyTLException("Malformed response received. Please try again.")

            result = [r["translatedText"] for r in results] if response_type == "text" else results # type: ignore
            
        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result
    
##-------------------start-of-googletl_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def googletl_translate_async(text:typing.Union[str, typing.Iterable[str]],
                                       target_lang:str = "en",
                                       override_previous_settings:bool = True,
                                       decorator:typing.Callable | None = None,
                                       logging_directory:str | None = None,
                                       response_type:typing.Literal["text", "raw"] | None = "text",
                                       semaphore:int | None = 15,
                                       translation_delay:float | None = None,
                                       format:typing.Literal["text", "html"] = "text",
                                       source_lang:str | None = None) -> typing.Union[typing.List[str], str, typing.List[typing.Any], typing.Any]:
        
        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for specific translation functions and will be ignored")
            logging_directory = None

        """

        Asynchronous version of googletl_translate().

        Translates the given text to the target language using Google Translate.
        Will generally be faster for iterables. Order is preserved.

        This function assumes that the credentials have already been set.

        It is unknown whether Google Translate has backoff retrying implemented. Assume it does not exist.

        Google Translate v2 API is poorly documented and type hints are near non-existent. typing.Any return types are used for the raw response type.

        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Google Translate function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        response_type (literal["text", "raw"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response.
        semaphore (int) : The number of concurrent requests to make. Default is 15.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        format (string or None) : The format of the text. Can be 'text' or 'html'. Default is 'text'. Google Translate appears to be able to translate html but this has not been tested thoroughly by EasyTL.
        source_lang (string or None) : The source language to translate from.

        Returns:
        result (string or list - string or any or list - any) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of any objects if the response type is 'raw' and input was an iterable, an any object otherwise.

        """

        assert response_type in ["text", "raw"], InvalidResponseFormatException("Invalid response type specified. Must be 'text' or 'raw'.")

        assert format in ["text", "html"], InvalidResponseFormatException("Invalid format specified. Must be 'text' or 'html'.")

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("google translate")

        if(override_previous_settings == True):
            GoogleTLService._set_attributes(target_language=target_lang, 
                                            format=format, 
                                            source_language=source_lang, 
                                            decorator=decorator, 
                                            semaphore=semaphore, 
                                            rate_limit_delay=translation_delay)
            
        ## This section may seem overly complex, but it is necessary to apply the decorator outside of the function call to avoid infinite recursion.
        ## Attempting to dynamically apply the decorator within the function leads to unexpected behavior, where this function's arguments are passed to the function instead of the intended translation function.
        def translate(text):
            return GoogleTLService._translate_text_async(text)
        
        if(decorator is not None):
            translate = GoogleTLService._decorator_to_use(GoogleTLService._translate_text_async) ## type: ignore

        else:
            translate = GoogleTLService._translate_text_async
            
        if(isinstance(text, str)):
            _result = await translate(text)

            assert not isinstance(_result, list), EasyTLException("Malformed response received. Please try again.")

            result = _result if response_type == "raw" else _result["translatedText"]
            
        elif(_is_iterable_of_strings(text)):
            _tasks = [translate(t) for t in text]
            _results = await asyncio.gather(*_tasks)
            
            assert isinstance(_results, list), EasyTLException("Malformed response received. Please try again.")

            result = [_r["translatedText"] for _r in _results] if response_type == "text" else _results # type: ignore
                
        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result
    
##-------------------start-of-deepl_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def deepl_translate(text:typing.Union[str, typing.Iterable[str]],
                        target_lang:str | Language = "EN-US",
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
                        logging_directory:str | None = None,
                        response_type:typing.Literal["text", "raw"] | None = "text",
                        translation_delay:float | None = None,
                        source_lang:str | Language | None = None,
                        context:str | None = None,
                        split_sentences:typing.Literal["OFF", "ALL", "NO_NEWLINES"] |  SplitSentences | None = "ALL",
                        preserve_formatting:bool | None = None,
                        formality:typing.Literal["default", "more", "less", "prefer_more", "prefer_less"] | Formality | None = None,
                        glossary:str | GlossaryInfo | None = None,
                        tag_handling:typing.Literal["xml", "html"] | None = None,
                        outline_detection:bool | None = None,
                        non_splitting_tags:str | typing.List[str] | None = None,
                        splitting_tags:str | typing.List[str] | None = None,
                        ignore_tags:str | typing.List[str] | None = None) -> typing.Union[typing.List[str], str, typing.List[TextResult], TextResult]:
        
        """

        Translates the given text to the target language using DeepL.

        This function assumes that the API key has already been set.

        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string or Language) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a DeepL translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying. If this parameter is None, DeepL will retry your request 5 times before failing. Otherwise, the given decorator will be used.
        response_type (literal["text", "raw"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a TextResult object.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        source_lang (string or Language or None) : The source language to translate from.
        context (string or None) : Additional information for the translator to be considered when translating. Not translated itself. This is a DeepL alpha feature and may be removed at any time.
        split_sentences (literal or SplitSentences or None) : How to split sentences.
        preserve_formatting (bool or None) : Whether to preserve formatting.
        formality (literal or Formality or None) : The formality level to use.
        glossary (string or GlossaryInfo or None) : The glossary to use.
        tag_handling (literal or None) : How to handle tags.
        outline_detection (bool or None) : Whether to detect outlines.
        non_splitting_tags (string or list or None) : Tags that should not be split.
        splitting_tags (string or list or None) : Tags that should be split.
        ignore_tags (string or list or None) : Tags that should be ignored.

        Returns:
        result (string or list - string or TextResult or list - TextResult) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of TextResult objects if the response type is 'raw' and input was an iterable, a TextResult object otherwise.

        """

        assert response_type in ["text", "raw"], InvalidResponseFormatException("Invalid response type specified. Must be 'text' or 'raw'.")

        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for specific translation functions and will be ignored")
            logging_directory = None

        EasyTL.test_credentials("deepl")

        if(override_previous_settings == True):
            DeepLService._set_attributes(target_lang = target_lang, 
                                        source_lang = source_lang, 
                                        context = context, 
                                        split_sentences = split_sentences,
                                        preserve_formatting = preserve_formatting, 
                                        formality = formality, 
                                        glossary = glossary, 
                                        tag_handling = tag_handling, 
                                        outline_detection = outline_detection, 
                                        non_splitting_tags = non_splitting_tags, 
                                        splitting_tags = splitting_tags, 
                                        ignore_tags = ignore_tags,
                                        decorator=decorator,
                                        semaphore=None,
                                        rate_limit_delay=translation_delay)
            
        ## This section may seem overly complex, but it is necessary to apply the decorator outside of the function call to avoid infinite recursion.
        ## Attempting to dynamically apply the decorator within the function leads to unexpected behavior, where this function's arguments are passed to the function instead of the intended translation function.
        def translate(text):
            return DeepLService._translate_text(text)
        
        if(decorator is not None):
            translate = DeepLService._decorator_to_use(DeepLService._translate_text) ## type: ignore

        else:
            translate = DeepLService._translate_text

        if(isinstance(text, str)):
            result = translate(text)
        
            assert not isinstance(result, list), EasyTLException("Malformed response received. Please try again.")

            result = result if response_type == "raw" else result.text
        
        elif(_is_iterable_of_strings(text)):

            results = [translate(t) for t in text]

            assert isinstance(results, list), EasyTLException("Malformed response received. Please try again.")

            result = [_r.text for _r in results] if response_type == "text" else results # type: ignore    
            
        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result  ## type: ignore
        
##-------------------start-of-deepl_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def deepl_translate_async(text:typing.Union[str, typing.Iterable[str]],
                            target_lang:str | Language = "EN-US",
                            override_previous_settings:bool = True,
                            decorator:typing.Callable | None = None,
                            logging_directory:str | None = None,
                            response_type:typing.Literal["text", "raw"] | None = "text",
                            semaphore:int | None = 15,
                            translation_delay:float | None = None,
                            source_lang:str | Language | None = None,
                            context:str | None = None,
                            split_sentences:typing.Literal["OFF", "ALL", "NO_NEWLINES"] |  SplitSentences | None = "ALL",
                            preserve_formatting:bool | None = None,
                            formality:typing.Literal["default", "more", "less", "prefer_more", "prefer_less"] | Formality | None = None,
                            glossary:str | GlossaryInfo | None = None,
                            tag_handling:typing.Literal["xml", "html"] | None = None,
                            outline_detection:bool | None = None,
                            non_splitting_tags:str | typing.List[str] | None = None,
                            splitting_tags:str | typing.List[str] | None = None,
                            ignore_tags:str | typing.List[str] | None = None) -> typing.Union[typing.List[str], str, typing.List[TextResult], TextResult]:

        """

        Asynchronous version of deepl_translate().
        
        Translates the given text to the target language using DeepL.

        Will generally be faster for iterables. Order is preserved.

        This function assumes that the API key has already been set.
        
        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string or Language) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a DeepL translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying. If this parameter is None, DeepL will retry your request 5 times before failing. Otherwise, the given decorator will be used.
        response_type (literal["text", "raw"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a TextResult object.
        semaphore (int) : The number of concurrent requests to make. Default is 15.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        source_lang (string or Language or None) : The source language to translate from.
        context (string or None) : Additional information for the translator to be considered when translating. Not translated itself. This is a DeepL alpha feature and may be removed at any time.
        split_sentences (literal or SplitSentences or None) : How to split sentences.
        preserve_formatting (bool or None) : Whether to preserve formatting.
        formality (literal or Formality or None) : The formality level to use.
        glossary (string or GlossaryInfo or None) : The glossary to use.
        tag_handling (literal or None) : How to handle tags.
        outline_detection (bool or None) : Whether to detect outlines.
        non_splitting_tags (string or list or None) : Tags that should not be split.
        splitting_tags (string or list or None) : Tags that should be split.
        ignore_tags (string or list or None) : Tags that should be ignored.

        Returns:
        result (string or list - string or TextResult or list - TextResult) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of TextResult objects if the response type is 'raw' and input was an iterable, a TextResult object otherwise.

        """

        assert response_type in ["text", "raw"], InvalidResponseFormatException("Invalid response type specified. Must be 'text' or 'raw'.")

        
        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for specific translation functions and will be ignored")
            logging_directory = None

        EasyTL.test_credentials("deepl")

        if(override_previous_settings == True):
            DeepLService._set_attributes(target_lang=target_lang, 
                                        source_lang=source_lang, 
                                        context=context, 
                                        split_sentences=split_sentences,
                                        preserve_formatting=preserve_formatting, 
                                        formality=formality, 
                                        glossary=glossary, 
                                        tag_handling=tag_handling, 
                                        outline_detection=outline_detection, 
                                        non_splitting_tags=non_splitting_tags, 
                                        splitting_tags=splitting_tags, 
                                        ignore_tags=ignore_tags,
                                        decorator=decorator,
                                        semaphore=semaphore,
                                        rate_limit_delay=translation_delay)
            
        ## This section may seem overly complex, but it is necessary to apply the decorator outside of the function call to avoid infinite recursion.
        ## Attempting to dynamically apply the decorator within the function leads to unexpected behavior, where this function's arguments are passed to the function instead of the intended translation function.
        def translate(text):
            return DeepLService._translate_text_async(text)
        
        if(decorator is not None):
            translate = DeepLService._decorator_to_use(DeepLService._translate_text_async) ## type: ignore

        else:
            translate = DeepLService._translate_text_async

        if(isinstance(text, str)):
            _result = await translate(text)

            assert not isinstance(_result, list), EasyTLException("Malformed response received. Please try again.")

            result = _result if response_type == "raw" else _result.text
            
        elif(_is_iterable_of_strings(text)):
            _tasks = [translate(t) for t in text]
            _results = await asyncio.gather(*_tasks)
            
            assert isinstance(_results, list), EasyTLException("Malformed response received. Please try again.")

            result = [_r.text for _r in _results] if response_type == "text" else _results # type: ignore
                
        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result
            
##-------------------start-of-gemini_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def gemini_translate(text:typing.Union[str, typing.Iterable[str]],
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
                        logging_directory:str | None = None,
                        response_type:typing.Literal["text", "raw", "json", "raw_json"] | None = "text",
                        response_schema:str | typing.Mapping[str, typing.Any] | None = None,
                        translation_delay:float | None = None,
                        translation_instructions:str | None = None,
                        model:str="gemini-pro",
                        temperature:float=0.5,
                        top_p:float=0.9,
                        top_k:int=40,
                        stop_sequences:typing.List[str] | None=None,
                        max_output_tokens:int | None=None,
                        stream:bool = False) -> typing.Union[typing.List[str], str, GenerateContentResponse, typing.List[GenerateContentResponse], 
                                                           typing.Iterator[GenerateContentResponse]]:
    
        """

        Translates the given text using Gemini.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        It is not known whether Gemini has backoff retrying implemented. Assume it does not exist.

        Streaming Support:
        - Streaming is only supported for single text inputs (not iterables)
        - When streaming is enabled (stream=True):
          - The response will be an iterator of GenerateContentResponse chunks
          - Each chunk contains a delta of the generated text
          - JSON mode and other response types are not supported with streaming
          - Typical usage is to iterate over chunks and print content as it arrives
        
        Example streaming usage:
        stream_response = EasyTL.gemini_translate("Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
                                                model="gemini-1.5-flash", 
                                                translation_instructions="Translate this to German. Take your time and translate word by word.", 
                                                stream=True,
                                                decorator=decorator)
        
        for chunk in stream_response: # type: ignore
            if hasattr(chunk, 'text') and chunk.text is not None: # type: ignore
                print(chunk.text, end="", flush=True) # type: ignore
                time.sleep(0.1)

        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Gemini translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        response_type (literal["text", "raw", "json", "raw_json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a GenerateContentResponse object, 'json' returns a json-parseable string. 'raw_json' returns the raw response, a GenerateContentResponse object, but with the content as a json-parseable string.
        response_schema (string or mapping or None) : The schema to use for the response. If None, no schema is used. This is only used if the response type is 'json' or 'json_raw'. EasyTL only validates the schema to the extend that it is None or a valid json. It does not validate the contents of the json.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        model (string) : The model to use. (E.g. 'gemini-pro' or 'gemini-pro-1.5-latest')
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        top_k (int) : The top k tokens to consider. Generally, alter this or temperature or top_p, not all three.
        stop_sequences (list or None) : String sequences that will cause the model to stop translating if encountered, generally useless.
        max_output_tokens (int or None) : The maximum number of tokens to output.
        stream (bool) : Whether to stream the response. If True, returns an iterator that yields chunks of the response as they become available.

        Returns:
        result (string or list - string or GenerateContentResponse or list - GenerateContentResponse or Iterator[GenerateContentResponse]) : 
            The translation result. A list of strings if the input was an iterable, a string otherwise. 
            A list of GenerateContentResponse objects if the response type is 'raw' and input was an iterable, a GenerateContentResponse object otherwise.
            An iterator of GenerateContentResponse chunks if streaming is enabled.

        """

        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for gemini_translate() and will be ignored")
            logging_directory = None

        assert response_type in ["text", "raw", "json", "raw_json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw', 'json' or 'raw_json'.")

        _settings = _return_curated_gemini_settings(locals())

        _validate_easytl_llm_translation_settings(_settings, "gemini")

        _validate_stop_sequences(stop_sequences)

        _validate_text_length(text, model, service="gemini")

        response_schema = _validate_response_schema(response_schema)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("gemini")

        json_mode = True if response_type in ["json", "raw_json"] else False

        if(stream):
            if(isinstance(text, typing.Iterable) and not isinstance(text, str)):
                raise ValueError("Streaming is only supported for single text inputs, not iterables")
            if(json_mode):
                raise ValueError("JSON mode is not supported with streaming")

        if(override_previous_settings == True):
            GeminiService._set_attributes(model=model,
                                          system_message=translation_instructions,
                                          temperature=temperature,
                                          top_p=top_p,
                                          top_k=top_k,
                                          candidate_count=1,
                                          stream=stream,
                                          stop_sequences=stop_sequences,
                                          max_output_tokens=max_output_tokens,
                                          decorator=decorator,
                                          semaphore=None,
                                          rate_limit_delay=translation_delay,
                                          json_mode=json_mode,
                                          response_schema=response_schema)
            
            ## Done afterwards, cause default translation instructions can change based on set_attributes()       
            GeminiService._system_message = translation_instructions or GeminiService._default_translation_instructions

        if(stream):
            return GeminiService._translate_text(text) ## type: ignore

        if(isinstance(text, str)):
            _result = GeminiService._translate_text(text)
            
            assert not isinstance(_result, list) and hasattr(_result, "text"), EasyTLException("Malformed response received. Please try again.")
            
            result = _result if response_type in ["raw", "raw_json"] else _result.text # type: ignore

        elif(_is_iterable_of_strings(text)):
            
            _results = [GeminiService._translate_text(t) for t in text]

            assert isinstance(_results, list) and all([hasattr(_r, "text") for _r in _results]), EasyTLException("Malformed response received. Please try again.")

            result = [_r.text for _r in _results] if response_type in ["text","json"] else _results # type: ignore
            
        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result

##-------------------start-of-gemini_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    async def gemini_translate_async(text:typing.Union[str, typing.Iterable[str]],
                                    override_previous_settings:bool = True,
                                    decorator:typing.Callable | None = None,
                                    logging_directory:str | None = None,
                                    response_type:typing.Literal["text", "raw", "json", "raw_json"] | None = "text",
                                    response_schema:str | typing.Mapping[str, typing.Any] | None = None,
                                    semaphore:int | None = 5,
                                    translation_delay:float | None = None,
                                    translation_instructions:str | None = None,
                                    model:str="gemini-pro",
                                    temperature:float=0.5,
                                    top_p:float=0.9,
                                    top_k:int=40,
                                    stop_sequences:typing.List[str] | None=None,
                                    max_output_tokens:int | None=None,
                                    stream:bool = False) -> typing.Union[typing.List[str], str, AsyncGenerateContentResponse, typing.List[AsyncGenerateContentResponse], 
                                                                       typing.AsyncIterator[AsyncGenerateContentResponse]]:
        
        """

        Asynchronous version of gemini_translate().
        Will generally be faster for iterables. Order is preserved.

        Translates the given text using Gemini.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        It is not known whether Gemini has backoff retrying implemented. Assume it does not exist.

        Streaming Support:
        - Streaming is only supported for single text inputs (not iterables)
        - When streaming is enabled (stream=True):
          - The response will be an async iterator of AsyncGenerateContentResponse chunks
          - Each chunk contains a delta of the generated text
          - JSON mode and other response types are not supported with streaming
          - Typical usage is to iterate over chunks and print content as it arrives
        
        Example streaming usage:
        async_stream_response = await EasyTL.gemini_translate_async(
            "Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
            model="gemini-1.5-flash", 
            translation_instructions="Translate this to German. Take your time and translate word by word.",
            stream=True,
            decorator=decorator
        )
        
        async for chunk in async_stream_response: # type: ignore
            if hasattr(chunk, 'text') and chunk.text is not None: # type: ignore
                print(chunk.text, end="", flush=True)
                await asyncio.sleep(0.1)
        
        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Gemini translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        response_type (literal["text", "raw", "json", "raw_json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, an AsyncGenerateContentResponse object, 'json' returns a json-parseable string. 'raw_json' returns the raw response, an AsyncGenerateContentResponse object, but with the content as a json-parseable string.
        response_schema (string or mapping or None) : The schema to use for the response. If None, no schema is used. This is only used if the response type is 'json' or 'json_raw'. EasyTL only validates the schema to the extend that it is None or a valid json. It does not validate the contents of the json.
        semaphore (int) : The number of concurrent requests to make. Default is 5 for 1.0 and 2 for 1.5 gemini models. For Gemini, it is recommend to use translation_delay along with the semaphore to prevent rate limiting.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        model (string) : The model to use. (E.g. 'gemini-pro' or 'gemini-pro-1.5-latest')
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        top_k (int) : The top k tokens to consider. Generally, alter this or temperature or top_p, not all three.
        stop_sequences (list or None) : String sequences that will cause the model to stop translating if encountered, generally useless.
        max_output_tokens (int or None) : The maximum number of tokens to output.
        stream (bool) : Whether to stream the response. If True, returns an async iterator that yields chunks of the response as they become available.

        Returns:
        result (string or list - string or AsyncGenerateContentResponse or list - AsyncGenerateContentResponse or AsyncIterator[AsyncGenerateContentResponse]) : 
            The translation result. A list of strings if the input was an iterable, a string otherwise. 
            A list of AsyncGenerateContentResponse objects if the response type is 'raw' and input was an iterable, an AsyncGenerateContentResponse object otherwise.
            An async iterator of AsyncGenerateContentResponse chunks if streaming is enabled.

        """

        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for gemini_translate_async() and will be ignored")
            logging_directory = None

        assert response_type in ["text", "raw", "json", "raw_json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw', 'json' or 'raw_json'.")

        _settings = _return_curated_gemini_settings(locals())

        _validate_easytl_llm_translation_settings(_settings, "gemini")

        _validate_stop_sequences(stop_sequences)

        _validate_text_length(text, model, service="gemini")

        response_schema = _validate_response_schema(response_schema)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("gemini")

        json_mode = True if response_type in ["json", "raw_json"] else False

        if(stream):
            if(isinstance(text, typing.Iterable) and not isinstance(text, str)):
                raise ValueError("Streaming is only supported for single text inputs, not iterables")
            if(json_mode):
                raise ValueError("JSON mode is not supported with streaming")

        if(override_previous_settings == True):
            GeminiService._set_attributes(model=model,
                                          system_message=translation_instructions,
                                          temperature=temperature,
                                          top_p=top_p,
                                          top_k=top_k,
                                          candidate_count=1,
                                          stream=stream,
                                          stop_sequences=stop_sequences,
                                          max_output_tokens=max_output_tokens,
                                          decorator=decorator,
                                          semaphore=semaphore,
                                          rate_limit_delay=translation_delay,
                                          json_mode=json_mode,
                                          response_schema=response_schema)
            
            ## Done afterwards, cause default translation instructions can change based on set_attributes()
            GeminiService._system_message = translation_instructions or GeminiService._default_translation_instructions
            
        if(stream):
            return await GeminiService._translate_text_async(text) ## type: ignore

        if(isinstance(text, str)):
            _result = await GeminiService._translate_text_async(text)

            result = _result if response_type in ["raw", "raw_json"] else _result.text
            
        elif(_is_iterable_of_strings(text)):
            _tasks = [GeminiService._translate_text_async(_t) for _t in text]
            _results = await asyncio.gather(*_tasks)

            result = [_r.text for _r in _results] if response_type in ["text","json"] else _results # type: ignore

        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result
            
##-------------------start-of-openai_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def openai_translate(text:typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]],
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
                        logging_directory:str | None = None,
                        response_type:typing.Literal["text", "raw", "json", "raw_json"] | None = "text",
                        response_schema:typing.Union[typing.Mapping[str, typing.Any], typing.Type[BaseModel], None] = None,
                        translation_delay:float | None = None,
                        translation_instructions:str | SystemTranslationMessage | None = None,
                        model:str="gpt-4",
                        temperature:float | None | NotGiven = NOT_GIVEN,
                        top_p:float | None | NotGiven = NOT_GIVEN,
                        stop:typing.List[str] | None | NotGiven = NOT_GIVEN,
                        max_tokens:int | None | NotGiven = NOT_GIVEN,
                        presence_penalty:float | None | NotGiven = NOT_GIVEN,
                        frequency_penalty:float | None | NotGiven = NOT_GIVEN,
                        stream:bool = False
                        ) -> typing.Union[typing.List[str], str, typing.List[ChatCompletion], ChatCompletion, 
                                        typing.Iterator[ChatCompletion], typing.AsyncIterator[ChatCompletion]]:
        
        """

        Translates the given text using OpenAI.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        Due to how OpenAI's API works, NOT_GIVEN is treated differently than None. If a parameter is set to NOT_GIVEN, it is not passed to the API. If it is set to None, it is passed to the API as None.
        
        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        For Structured Responses, you can pass in a pydantic base model for the API to follow when returning the response. 
        
        EasyTL follows the traditional schema approach as well for OpenAI Structured Responses.
        https://platform.openai.com/docs/guides/structured-outputs/how-to-use?context=without_parse&lang=python

        It'll look more or less like:
        ```python
        "name": "convert_to_json",
        "schema": {
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
        ```

        Streaming Support:
        - Streaming is only supported for single text inputs (not iterables)
        - When streaming is enabled (stream=True):
          - The response will be an iterator of ChatCompletion chunks
          - Each chunk contains a delta of the generated text
          - JSON mode and other response types are not supported with streaming
          - Typical usage is to iterate over chunks and print content as it arrives
        
        Example streaming usage ():
        stream_response = EasyTL.openai_translate("Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
                                                model="gpt-4", 
                                                translation_instructions="Translate this to German. Take your time and translate word by word.", 
                                                stream=True,
                                                decorator=decorator)
        
        for chunk in stream_response: # type: ignore
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
                time.sleep(0.1)

        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to an OpenAI translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying. If this is None, OpenAI will retry the request twice if it fails.
        response_type (literal["text", "raw", "json", "raw_json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a ChatCompletion object, 'json' returns a json-parseable string. 'raw_json' returns the raw response, a ChatCompletion object, but with the content as a json-parseable string.
        response_schema (mapping or BaseModel or None) : The schema to use for the response. If None, no schema is used. This is only used if the response type is 'json' or 'json_raw'. EasyTL only validates the schema to the extend that it is None or a valid json. It does not validate the contents of the json.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or SystemTranslationMessage or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        model (string) : The model to use. (E.g. 'gpt-4', 'gpt-3.5-turbo-0125', 'gpt-4o', etc.)
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        stop (list or None) : String sequences that will cause the model to stop translating if encountered, generally useless.
        max_tokens (int or None) : The maximum number of tokens to output.
        presence_penalty (float) : The presence penalty to use. This penalizes the model from repeating the same content in the output. Shouldn't be messed with for translation.
        frequency_penalty (float) : The frequency penalty to use. This penalizes the model from using the same words too frequently in the output. Shouldn't be messed with for translation.
        stream (bool) : Whether to stream the response. If True, returns an iterator that yields chunks of the response as they become available.

        Returns:
        result (string or list - string or ChatCompletion or list - ChatCompletion or Iterator[ChatCompletion] or AsyncIterator[ChatCompletion]) : 
            The translation result. A list of strings if the input was an iterable, a string otherwise. 
            A list of ChatCompletion objects if the response type is 'raw' and input was an iterable, a ChatCompletion object otherwise.
            An iterator of ChatCompletion chunks if streaming is enabled.
        """

        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for openai_translate() and will be ignored")
            logging_directory = None

        assert response_type in ["text", "raw", "json", "raw_json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw', 'json' or 'raw_json'.")

        _settings = _return_curated_openai_settings(locals())

        _validate_easytl_llm_translation_settings(_settings, "openai")

        _validate_stop_sequences(stop)

        _validate_text_length(text, model, service="openai")

        if not (isinstance(response_schema, type) and issubclass(response_schema, BaseModel)):
            response_schema = _validate_response_schema(response_schema)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("openai")

        json_mode = True if response_type in ["json", "raw_json"] else False
        
        if(override_previous_settings == True):
            OpenAIService._set_attributes(model=model,
                                        temperature=temperature,
                                        logit_bias=None,
                                        top_p=top_p,
                                        n=1,
                                        stream=stream,
                                        stop=stop,
                                        max_tokens=max_tokens,
                                        presence_penalty=presence_penalty,
                                        frequency_penalty=frequency_penalty,
                                        decorator=decorator,
                                        semaphore=None,
                                        rate_limit_delay=translation_delay,
                                        json_mode=json_mode,
                                        response_schema=response_schema)

            ## Done afterwards, cause default translation instructions can change based on set_attributes()
            translation_instructions = translation_instructions or OpenAIService._default_translation_instructions
        
        else:
            translation_instructions = OpenAIService._system_message

        assert isinstance(text, str) or _is_iterable_of_strings(text) or isinstance(text, ModelTranslationMessage) or _is_iterable_of_strings(text), InvalidTextInputException("text must be a string, an iterable of strings, a ModelTranslationMessage or an iterable of ModelTranslationMessages.")

        translation_batches = OpenAIService._build_translation_batches(text, translation_instructions)
        
        if(stream):
            if(len(translation_batches) > 1):
                raise ValueError("Streaming is only supported for single text inputs, not iterables")
            return OpenAIService._translate_text(translation_batches[0][1], translation_batches[0][0])
        
        translations = []
        
        for _text, _translation_instructions in translation_batches:
            _result = OpenAIService._translate_text(_translation_instructions, _text)

            ## Skip validation for streaming responses
            if(stream):
                translations.append(_result)
                continue

            ## Validate normal responses
            if(not hasattr(_result, "choices")):
                raise EasyTLException("Malformed response received. Please try again.")

            translation = _result if response_type in ["raw", "raw_json"] else _result.choices[0].message.content # type: ignore
            
            translations.append(translation)
        
        result = translations if isinstance(text, typing.Iterable) and not isinstance(text, str) else translations[0]
        
        return result
    
##-------------------start-of-openai_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def openai_translate_async(text:typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]],
                                    override_previous_settings:bool = True,
                                    decorator:typing.Callable | None = None,
                                    logging_directory:str | None = None,
                                    response_type:typing.Literal["text", "raw", "json", "raw_json"] | None = "text",
                                    response_schema:typing.Union[typing.Mapping[str, typing.Any], typing.Type[BaseModel], None] = None,
                                    semaphore:int | None = 5,
                                    translation_delay:float | None = None,
                                    translation_instructions:str | SystemTranslationMessage | None = None,
                                    model:str="gpt-4",
                                    temperature:float | None | NotGiven = NOT_GIVEN,
                                    top_p:float | None | NotGiven = NOT_GIVEN,
                                    stop:typing.List[str] | None | NotGiven = NOT_GIVEN,
                                    max_tokens:int | None | NotGiven = NOT_GIVEN,
                                    presence_penalty:float | None | NotGiven = NOT_GIVEN,
                                    frequency_penalty:float | None | NotGiven = NOT_GIVEN,
                                    stream:bool = False
                                    ) -> typing.Union[typing.List[str], str, typing.List[ChatCompletion], ChatCompletion, 
                                                    typing.Iterator[ChatCompletion], typing.AsyncIterator[ChatCompletion]]:
        
        """

        Asynchronous version of openai_translate().
        Will generally be faster for iterables. Order is preserved.

        Translates the given text using OpenAI.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        Due to how OpenAI's API works, NOT_GIVEN is treated differently than None. If a parameter is set to NOT_GIVEN, it is not passed to the API. If it is set to None, it is passed to the API as None.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        For Structured Responses, you can pass in a pydantic base model for the API to follow when returning the response. 
        
        EasyTL follows the traditional schema approach as well for OpenAI Structured Responses.
        https://platform.openai.com/docs/guides/structured-outputs/how-to-use?context=without_parse&lang=python

        It'll look more or less like:
        ```python
        "name": "convert_to_json",
        "schema": {
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
        ```

        Streaming Support:
        - Streaming is only supported for single text inputs (not iterables)
        - When streaming is enabled (stream=True):
          - The response will be an iterator of ChatCompletion chunks
          - Each chunk contains a delta of the generated text
          - JSON mode and other response types are not supported with streaming
          - Typical usage is to iterate over chunks and print content as it arrives
        
        Example streaming usage:
        async_stream_response = await EasyTL.openai_translate_async(
            "Hello, world! This is a longer message that we'll see stream in real time. Each word should appear one by one.", 
            model="gpt-4", 
            translation_instructions="Translate this to German. Take your time and translate word by word.",
            stream=True,
            decorator=decorator
        )
        
        async for chunk in async_stream_response: # type: ignore
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
                await asyncio.sleep(0.1)

        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to an OpenAI translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying. If this is None, OpenAI will retry the request twice if it fails.
        response_type (literal["text", "raw", "json", "raw_json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a ChatCompletion object, 'json' returns a json-parseable string. 'raw_json' returns the raw response, a ChatCompletion object, but with the content as a json-parseable string.
        response_schema (mapping or BaseModel or None) : The schema to use for the response. If None, no schema is used. This is only used if the response type is 'json' or 'json_raw'. EasyTL only validates the schema to the extend that it is None or a valid json. It does not validate the contents of the json.
        semaphore (int) : The number of concurrent requests to make. Default is 5.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or SystemTranslationMessage or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        model (string) : The model to use. (E.g. 'gpt-4', 'gpt-3.5-turbo-0125', 'gpt-4o', etc.)
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        stop (list or None) : String sequences that will cause the model to stop translating if encountered, generally useless.
        max_tokens (int or None) : The maximum number of tokens to output.
        presence_penalty (float) : The presence penalty to use. This penalizes the model from repeating the same content in the output. Shouldn't be messed with for translation.
        frequency_penalty (float) : The frequency penalty to use. This penalizes the model from using the same words too frequently in the output. Shouldn't be messed with for translation.
        stream (bool) : Whether to stream the response. If True, returns an iterator that yields chunks of the response as they become available.

        Returns:
        result (string or list - string or ChatCompletion or list - ChatCompletion or Iterator[ChatCompletion] or AsyncIterator[ChatCompletion]) : 
            The translation result. A list of strings if the input was an iterable, a string otherwise. 
            A list of ChatCompletion objects if the response type is 'raw' and input was an iterable, a ChatCompletion object otherwise.
            An iterator of ChatCompletion chunks if streaming is enabled.
        """

        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for openai_translate_async() and will be ignored")
            logging_directory = None

        assert response_type in ["text", "raw", "json", "raw_json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw', 'json' or 'raw_json'.")

        _settings = _return_curated_openai_settings(locals())

        _validate_easytl_llm_translation_settings(_settings, "openai")

        _validate_stop_sequences(stop)

        _validate_text_length(text, model, service="openai")

        if not (isinstance(response_schema, type) and issubclass(response_schema, BaseModel)):
            response_schema = _validate_response_schema(response_schema)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("openai")

        json_mode = True if response_type in ["json", "raw_json"] else False

        if(override_previous_settings == True):
            OpenAIService._set_attributes(model=model,
                                        temperature=temperature,
                                        logit_bias=None,
                                        top_p=top_p,
                                        n=1,
                                        stream=stream,
                                        stop=stop,
                                        max_tokens=max_tokens,
                                        presence_penalty=presence_penalty,
                                        frequency_penalty=frequency_penalty,
                                        decorator=decorator,
                                        semaphore=semaphore,
                                        rate_limit_delay=translation_delay,
                                        json_mode=json_mode,
                                        response_schema=response_schema)
            
            ## Done afterwards, cause default translation instructions can change based on set_attributes()
            translation_instructions = translation_instructions or OpenAIService._default_translation_instructions

        else:
            translation_instructions = OpenAIService._system_message        
            
        assert isinstance(text, str) or _is_iterable_of_strings(text) or isinstance(text, ModelTranslationMessage) or _is_iterable_of_strings(text), InvalidTextInputException("text must be a string, an iterable of strings, a ModelTranslationMessage or an iterable of ModelTranslationMessages.")

        _translation_batches = OpenAIService._build_translation_batches(text, translation_instructions)


        if(stream):
            if(len(_translation_batches) > 1):
                raise ValueError("Streaming is only supported for single text inputs, not iterables")
            return await OpenAIService._translate_text_async(_translation_batches[0][1], _translation_batches[0][0])

        _translation_tasks = []

        for _text, _translation_instructions in _translation_batches:
            _task = OpenAIService._translate_text_async(_translation_instructions, _text)
            _translation_tasks.append(_task)

        _results = await asyncio.gather(*_translation_tasks)

        _results:typing.List[ChatCompletion] = _results

        assert all([hasattr(_r, "choices") for _r in _results]), EasyTLException("Malformed response received. Please try again.")

        translation = _results if response_type in ["raw","raw_json"] else [result.choices[0].message.content for result in _results if result.choices[0].message.content is not None]

        result = translation if isinstance(text, typing.Iterable) and not isinstance(text, str) else translation[0]

        return result # type: ignore
    
##-------------------start-of-anthropic_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def anthropic_translate(text:typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]],
                            override_previous_settings:bool = True,
                            decorator:typing.Callable | None = None,
                            logging_directory:str | None = None,
                            response_type:typing.Literal["text", "raw", "json", "raw_json"] | None = "text",
                            response_schema:str | typing.Mapping[str, typing.Any] | None = None,
                            translation_delay:float | None = None,
                            translation_instructions:str | None = None,
                            model:str="claude-3-haiku-20240307",
                            temperature:float | NotGiven = NOT_GIVEN,
                            top_p:float | NotGiven = NOT_GIVEN,
                            top_k:int | NotGiven = NOT_GIVEN,
                            stop_sequences:typing.List[str] | NotGiven = NOT_GIVEN,
                            max_output_tokens:int | NotGiven = NOT_GIVEN,
                            stream:bool = False) -> typing.Union[typing.List[str], str, 
                                                                  AnthropicMessage, typing.List[AnthropicMessage],
                                                                  typing.Iterator[AnthropicMessage], typing.AsyncIterator[AnthropicMessage]]:
        
        """

        Translates the given text using Anthropic.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        Due to how Anthropic's API works, NOT_GIVEN is treated differently than None. If a parameter is set to NOT_GIVEN, it is not passed to the API. 

        Anthropic's JSON response is quite unsophisticated, it costs a lot of extra tokens to return a json response. It's also inconsistent. Be careful when using it.

        Streaming Support:
        - Streaming is only supported for single text inputs (not iterables)
        - When streaming is enabled (stream=True):
          - The response will be an iterator of AnthropicMessage chunks
          - Each chunk contains a delta of the generated text
          - JSON mode and other response types are not supported with streaming
          - Typical usage is to iterate over chunks and print content as it arrives
        
        Example streaming usage:
        stream_response = EasyTL.anthropic_translate("Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
                                                   model="claude-3-haiku-20240307", 
                                                   translation_instructions="Translate this to German. Take your time and translate word by word.", 
                                                   stream=True,
                                                   decorator=decorator)
        
        for event in stream_response: # type: ignore
            if event.type == "content_block_delta":
                if hasattr(event.delta, 'text'):
                    print(event.delta.text, end="", flush=True)
                    time.sleep(0.1)
            elif event.type == "message_delta":
                if hasattr(event.delta, 'text'):
                    print(event.delta.text, end="", flush=True)
                    time.sleep(0.1)
            elif event.type == "message_stop":
                print("\nTranslation completed.")

        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to an Anthropic translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying. If this is None, Anthropic will retry the request twice if it fails.
        response_type (literal["text", "raw", "json", "raw_json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, an AnthropicMessage object, 'json' returns a json-parseable string. 'raw_json' returns the raw response, an AnthropicMessage object, but with the content as a json-parseable string.
        response_schema (string or mapping or None) : The schema to use for the response. If None, no schema is used. This is only used if the response type is 'json' or 'json_raw'. EasyTL only validates the schema to the extend that it is None or a valid json. It does not validate the contents of the json. 
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or SystemTranslationMessage or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        model (string) : The model to use. (E.g. 'claude-3-haiku-20240307', 'claude-3-sonnet-20240229' or 'claude-3-opus-20240229')
        temperature (float or NotGiven) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float or NotGiven) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        top_k (int or NotGiven) : The top k tokens to consider. Generally, alter this or temperature or top_p, not all three.
        stop_sequences (list or NotGiven) : String sequences that will cause the model to stop translating if encountered, generally useless.
        max_output_tokens (int or NotGiven) : The maximum number of tokens to output.
        stream (bool) : Whether to stream the response. If True, returns an iterator that yields chunks of the response as they become available.

        Returns:
        result (string or list - string or AnthropicMessage or list - AnthropicMessage or Iterator[AnthropicMessage] or AsyncIterator[AnthropicMessage]) : 
            The translation result. A list of strings if the input was an iterable, a string otherwise. 
            A list of AnthropicMessage objects if the response type is 'raw' and input was an iterable, an AnthropicMessage object otherwise.
            An iterator of AnthropicMessage chunks if streaming is enabled.
        """

        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for anthropic_translate() and will be ignored")
            logging_directory = None

        assert response_type in ["text", "raw", "json", "raw_json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw', 'json' or 'raw_json'.")

        _settings = _return_curated_anthropic_settings(locals())

        _validate_easytl_llm_translation_settings(_settings, "anthropic")

        _validate_stop_sequences(stop_sequences)

        _validate_text_length(text, model, service="anthropic")

        response_schema = _validate_response_schema(response_schema)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("anthropic")

        json_mode = True if response_type in ["json", "raw_json"] else False

        if(stream):
            if(isinstance(text, typing.Iterable) and not isinstance(text, str)):
                raise ValueError("Streaming is only supported for single text inputs, not iterables")
            if(json_mode):
                raise ValueError("JSON mode is not supported with streaming")

        if(override_previous_settings == True):
            AnthropicService._set_attributes(model=model,
                                            system=translation_instructions,
                                            temperature=temperature,
                                            top_p=top_p,
                                            top_k=top_k,
                                            stop_sequences=stop_sequences,
                                            stream=stream,
                                            max_tokens=max_output_tokens,
                                            decorator=decorator,
                                            semaphore=None,
                                            rate_limit_delay=translation_delay,
                                            json_mode=json_mode,
                                            response_schema=response_schema)
            
            ## Done afterwards, cause default translation instructions can change based on set_attributes()
            AnthropicService._system = translation_instructions or AnthropicService._default_translation_instructions

        assert isinstance(text, str) or _is_iterable_of_strings(text) or isinstance(text, ModelTranslationMessage) or _is_iterable_of_strings(text), InvalidTextInputException("text must be a string, an iterable of strings, a ModelTranslationMessage or an iterable of ModelTranslationMessages.")

        _translation_batches = AnthropicService._build_translation_batches(text)

        if(stream):
            return AnthropicService._translate_text(AnthropicService._system, _translation_batches[0])

        _translations = []

        for _text in _translation_batches:

            _result = AnthropicService._translate_text(AnthropicService._system, _text)

            assert not isinstance(_result, list) and hasattr(_result, "content"), EasyTLException("Malformed response received. Please try again.")

            if(response_type in ["raw", "raw_json"]):
                translation = _result

            ## response structure is inconsistent, so we have to check for both types of responses
            else:
                content = _result.content # type: ignore

                if(isinstance(content[0], AnthropicTextBlock)):
                    translation = content[0].text # type: ignore

                elif(isinstance(content[0], AnthropicToolUseBlock)):
                    translation = content[0].input # type: ignore
                            
            _translations.append(translation)

        ## If originally a single text was provided, return a single translation instead of a list
        result = _translations if isinstance(text, typing.Iterable) and not isinstance(text, str) else _translations[0]

        return result
    
##-------------------start-of-anthropic_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def anthropic_translate_async(text:typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]],
                                        override_previous_settings:bool = True,
                                        decorator:typing.Callable | None = None,
                                        logging_directory:str | None = None,
                                        response_type:typing.Literal["text", "raw", "json", "raw_json"] | None = "text",
                                        response_schema:str | typing.Mapping[str, typing.Any] | None = None,
                                        semaphore:int | None = 5,
                                        translation_delay:float | None = None,
                                        translation_instructions:str | None = None,
                                        model:str="claude-3-haiku-20240307",
                                        temperature:float | NotGiven = NOT_GIVEN,
                                        top_p:float | NotGiven = NOT_GIVEN,
                                        top_k:int | NotGiven = NOT_GIVEN,
                                        stop_sequences:typing.List[str] | NotGiven = NOT_GIVEN,
                                        max_output_tokens:int | NotGiven = NOT_GIVEN,
                                        stream:bool = False) -> typing.Union[typing.List[str], str, 
                                                                          AnthropicMessage, typing.List[AnthropicMessage],
                                                                          typing.Iterator[AnthropicMessage], typing.AsyncIterator[AnthropicMessage]]:

        """

        Asynchronous version of anthropic_translate().
        Will generally be faster for iterables. Order is preserved.

        Translates the given text using Anthropic.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        Due to how Anthropic's API works, NOT_GIVEN is treated differently than None. If a parameter is set to NOT_GIVEN, it is not passed to the API.

        Anthropic's JSON response is quite unsophisticated, it costs a lot of extra tokens to return a json response. It's also inconsistent. Be careful when using it.

        Streaming Support:
        - Streaming is only supported for single text inputs (not iterables)
        - When streaming is enabled (stream=True):
          - The response will be an async iterator of AnthropicMessage chunks
          - Each chunk contains a delta of the generated text
          - JSON mode and other response types are not supported with streaming
          - Typical usage is to iterate over chunks and print content as it arrives
        
        Example streaming usage:
        async_stream_response = await EasyTL.anthropic_translate_async(
            "Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
            model="claude-3-haiku-20240307", 
            translation_instructions="Translate this to German. Take your time and translate word by word.",
            stream=True,
            decorator=decorator
        )
        
        async for event in async_stream_response: # type: ignore
            if event.type == "content_block_delta":
                if hasattr(event.delta, 'text'):
                    print(event.delta.text, end="", flush=True)
                    await asyncio.sleep(0.1)
            elif event.type == "message_delta":
                if hasattr(event.delta, 'text'):
                    print(event.delta.text, end="", flush=True)
                    await asyncio.sleep(0.1)
            elif event.type == "message_stop":
                print("\nTranslation completed.")

        Parameters:
        text (string | ModelTranslationMessage or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to an Anthropic translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying. If this is None, Anthropic will retry the request twice if it fails.
        response_type (literal["text", "raw", "json", "raw_json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, an AnthropicMessage object, 'json' returns a json-parseable string. 'raw_json' returns the raw response, an AnthropicMessage object, but with the content as a json-parseable string.
        response_schema (string or mapping or None) : The schema to use for the response. If None, no schema is used. This is only used if the response type is 'json' or 'json_raw'. EasyTL only validates the schema to the extend that it is None or a valid json. It does not validate the contents of the json.
        semaphore (int) : The number of concurrent requests to make. Default is 5.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or SystemTranslationMessage or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        model (string) : The model to use. (E.g. 'claude-3-haiku-20240307', 'claude-3-sonnet-20240229' or 'claude-3-opus-20240229')
        temperature (float or NotGiven) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float or NotGiven) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        top_k (int or NotGiven) : The top k tokens to consider. Generally, alter this or temperature or top_p, not all three.
        stop_sequences (list or NotGiven) : String sequences that will cause the model to stop translating if encountered, generally useless.
        max_output_tokens (int or NotGiven) : The maximum number of tokens to output.
        stream (bool) : Whether to stream the response. If True, returns an async iterator that yields chunks of the response as they become available.

        Returns:
        result (string or list - string or AnthropicMessage or list - AnthropicMessage or Iterator[AnthropicMessage] or AsyncIterator[AnthropicMessage]) : 
            The translation result. A list of strings if the input was an iterable, a string otherwise. 
            A list of AnthropicMessage objects if the response type is 'raw' and input was an iterable, an AnthropicMessage object otherwise.
            An async iterator of AnthropicMessage chunks if streaming is enabled.
        """

        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for anthropic_translate_async() and will be ignored")
            logging_directory = None

        assert response_type in ["text", "raw", "json", "raw_json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw', 'json' or 'raw_json'.")

        _settings = _return_curated_anthropic_settings(locals())

        _validate_easytl_llm_translation_settings(_settings, "anthropic")

        _validate_stop_sequences(stop_sequences)

        _validate_text_length(text, model, service="anthropic")

        response_schema = _validate_response_schema(response_schema)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("anthropic")

        json_mode = True if response_type in ["json", "raw_json"] else False

        if(stream):
            if(isinstance(text, typing.Iterable) and not isinstance(text, str)):
                raise ValueError("Streaming is only supported for single text inputs, not iterables")
            if(json_mode):
                raise ValueError("JSON mode is not supported with streaming")

        if(override_previous_settings == True):
            AnthropicService._set_attributes(model=model,
                                            system=translation_instructions,
                                            temperature=temperature,
                                            top_p=top_p,
                                            top_k=top_k,
                                            stop_sequences=stop_sequences,
                                            stream=stream,
                                            max_tokens=max_output_tokens,
                                            decorator=decorator,
                                            semaphore=semaphore,
                                            rate_limit_delay=translation_delay,
                                            json_mode=json_mode,
                                            response_schema=response_schema)
            
            ## Done afterwards, cause default translation instructions can change based on set_attributes()
            AnthropicService._system = translation_instructions or AnthropicService._default_translation_instructions
        
        assert isinstance(text, str) or _is_iterable_of_strings(text) or isinstance(text, ModelTranslationMessage) or _is_iterable_of_strings(text), InvalidTextInputException("text must be a string, an iterable of strings, a ModelTranslationMessage or an iterable of ModelTranslationMessages.")

        _translation_batches = AnthropicService._build_translation_batches(text)

        if(stream):
            if(len(_translation_batches) > 1):
                raise ValueError("Streaming is only supported for single text inputs, not iterables")
            return await AnthropicService._translate_text_async(translation_instructions, _translation_batches[0])

        _translation_tasks = []

        for _text in _translation_batches:
            _task = AnthropicService._translate_text_async(translation_instructions, _text)
            _translation_tasks.append(_task)

        _results = await asyncio.gather(*_translation_tasks)

        _results:typing.List[AnthropicMessage] = _results

        assert all([hasattr(_r, "content") for _r in _results]), EasyTLException("Malformed response received. Please try again.")

        if(response_type in ["raw", "raw_json"]):
            translations = _results

        ## response structure is different for beta
        else:
            translations = [result.content[0].input if isinstance(result.content[0], AnthropicToolUseBlock) else result.content[0].text for result in _results]
        
        result = translations if isinstance(text, typing.Iterable) and not isinstance(text, str) else translations[0]

        return result # type: ignore
    
##-------------------start-of-azure_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def azure_translate(text: typing.Union[str, typing.Iterable[str]],
                        target_lang:str = 'en',
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
                        logging_directory:str | None = None,
                        response_type:typing.Literal["text", "json"] | None = "text",
                        translation_delay:float | None = None,
                        api_version:str = '3.0',
                        azure_region:str = "global",
                        azure_endpoint:str = "https://api.cognitive.microsofttranslator.com/",
                        source_lang:str | None = None) -> typing.Union[typing.List[str], str]:
        
        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for specific translation functions and will be ignored")
            logging_directory = None

        """

        Translates the given text to the target language using Azure.

        This function assumes that the API key has already been set.

        It is unknown whether Azure Translate has backoff retrying implemented. Assume it does not  exist

        Default api_version, azure_region, and azure_endpoint values should be fine for most users.
        
        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string) : The target language for translation. Default is 'en'. These are ISO 639-1 language codes
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to an Azure translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying. 
        response_type (literal["text", "json"]) : The type of response to return. 'text' returns the translated text, 'json' returns the original response in json format.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        api_version (string) : The version of the Azure Translator API. Default is '3.0'.
        azure_region (string) : The Azure region to use for translation. Default is 'global'.
        azure_endpoint (string) : The Azure Translator API endpoint. Default is 'https://api.cognitive.microsofttranslator.com/'.
        source_lang (string or None) : The source language of the text. If None, the service will attempt to detect the language.

        Returns:
        result (string or list - string) : The translation result. A list of strings if the input was an iterable, a string otherwise.

        """

        assert response_type in ["text", "json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text' or 'json'.")


        EasyTL.test_credentials("azure", azure_region=azure_region)

        if(override_previous_settings == True):
            AzureService._set_attributes(target_language=target_lang,
                                        api_version=api_version,
                                        azure_region=azure_region,
                                        azure_endpoint=azure_endpoint,
                                        source_language=source_lang,
                                        decorator=decorator,
                                        semaphore=None,
                                        rate_limit_delay=translation_delay)
            
        ## This section may seem overly complex, but it is necessary to apply the decorator outside of the function call to avoid infinite recursion.
        ## Attempting to dynamically apply the decorator within the function leads to unexpected behavior, where this function's arguments are passed to the function instead of the intended translation function.

        def translate(text):
            return AzureService._translate_text(text)
        
        if(decorator is not None):
            translate = AzureService._decorator_to_use(AzureService._translate_text) # type: ignore

        else:
            translate = AzureService._translate_text

        if(isinstance(text, str)):

            result = translate(text)[0]

            assert not isinstance(result, list), EasyTLException("Malformed response received. Please try again.")

            result = result if response_type == "json" else result['translations'][0]['text']
        
        elif(_is_iterable_of_strings(text)):

            _results = [translate(_t) for _t in text]

            assert isinstance(_results, list), EasyTLException("Malformed response received. Please try again.")

            result = _results if response_type == "json" else [result[0]['translations'][0]['text'] for result in _results]

        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")

        return result # type: ignore

##-------------------start-of-azure_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def azure_translate_async(text: typing.Union[str, typing.Iterable[str]],
                                    target_lang:str = 'en',
                                    override_previous_settings:bool = True,
                                    decorator:typing.Callable | None = None,
                                    logging_directory:str | None = None,
                                    response_type:typing.Literal["text", "json"] | None = "text",
                                    semaphore:int | None = 15,
                                    translation_delay:float | None = None,
                                    api_version:str = '3.0',
                                    azure_region:str = "global",
                                    azure_endpoint:str = "https://api.cognitive.microsofttranslator.com/",
                                    source_lang:str | None = None) -> typing.Union[typing.List[str], str]:
        """

        Asynchronous version of azure_translate().
        Will generally be faster for iterables. Order is preserved.

        Translates the given text to the target language using Azure.

        This function assumes that the API key has already been set.

        It is unknown whether Azure Translate has backoff retrying implemented. Assume it does not  exist

        Default api_version, azure_region, and azure_endpoint values should be fine for most users.

        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string) : The target language for translation. Default is 'en'. These are ISO 639-1 language codes
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to an Azure translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        response_type (literal["text", "json"]) : The type of response to return. 'text' returns the translated text, 'json' returns the original response in json format.
        semaphore (int) : The number of concurrent requests to make. Default is 15.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        api_version (string) : The version of the Azure Translator API. Default is '3.0'.
        azure_region (string) : The Azure region to use for translation. Default is 'global'.
        azure_endpoint (string) : The Azure Translator API endpoint. Default is 'https://api.cognitive.microsofttranslator.com/'.
        source_lang (string or None) : The source language of the text. If None, the service will attempt to detect the language.

        Returns:
        result (string or list - string) : The translation result. A list of strings if the input was an iterable, a string otherwise.
        
        """

        if(logging_directory is not None):
            print("Warning: logging_directory parameter is deprecated for specific translation functions and will be ignored")
            logging_directory = None

        assert response_type in ["text", "json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text' or 'json'.")

        EasyTL.test_credentials("azure", azure_region=azure_region)

        if(override_previous_settings == True):
            AzureService._set_attributes(target_language=target_lang,
                                        api_version=api_version,
                                        azure_region=azure_region,
                                        azure_endpoint=azure_endpoint,
                                        source_language=source_lang,
                                        decorator=decorator,
                                        semaphore=semaphore,
                                        rate_limit_delay=translation_delay)
            
        ## This section may seem overly complex, but it is necessary to apply the decorator outside of the function call to avoid infinite recursion.
        ## Attempting to dynamically apply the decorator within the function leads to unexpected behavior, where this function's arguments are passed to the function instead of the intended translation function.

        async def translate(text):
            return await AzureService._translate_text_async(text)
        
        if(decorator is not None):
            translate = AzureService._decorator_to_use(AzureService._translate_text_async) # type: ignore

        else:
            translate = AzureService._translate_text_async

        if(isinstance(text, str)):

            result = (await translate(text))[0]

            assert not isinstance(result, list), EasyTLException("Malformed response received. Please try again.")

            result = result if response_type == "json" else result['translations'][0]['text']

        elif(_is_iterable_of_strings(text)):

            _tasks = [translate(_t) for _t in text]
            _results = await asyncio.gather(*_tasks)

            assert isinstance(_results, list), EasyTLException("Malformed response received. Please try again.")

            result = _results if response_type == "json" else [result[0]['translations'][0]['text'] for result in _results]

        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result # type: ignore

##-------------------start-of-translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def translate(text:str | typing.Iterable[str],
                  service:typing.Optional[typing.Literal["deepl", "openai", "gemini", "google translate", "anthropic", "azure"]] = "deepl",
                  **kwargs) -> typing.Union[typing.List[str], str, 
                                            typing.List[TextResult], TextResult, 
                                            typing.List[ChatCompletion], ChatCompletion,
                                            typing.List[GenerateContentResponse], GenerateContentResponse, 
                                            typing.List[typing.Any], typing.Any,
                                            typing.List[AnthropicMessage], AnthropicMessage]:
        
        """

        Translates the given text to the target language using the specified service.

        Please see the documentation for the specific translation function for the service you want to use.

        DeepL: deepl_translate() 
        OpenAI: openai_translate() 
        Gemini: gemini_translate() 
        Google Translate: googletl_translate() 
        Anthropic: anthropic_translate()
        Azure: azure_translate()

        All functions can return a list of strings or a string, depending on the input. The response type can be specified to return the raw response instead:
        DeepL: TextResult
        OpenAI: ChatCompletion
        Gemini: GenerateContentResponse
        Google Translate: any
        Anthropic: AnthropicMessage or AnthropicToolsBetaMessage
        Azure: str

        Parameters:
        service (string) : The service to use for translation.
        text (string) : The text to translate.
        **kwargs : The keyword arguments to pass to the translation function.

        Returns:
        result (string or list - string or TextResult or list - TextResult or GenerateContentResponse or list - GenerateContentResponse or ChatCompletion or list - ChatCompletion or any or list - any or AnthropicMessage or list - AnthropicMessage or AnthropicToolsBetaMessage or list - AnthropicToolsBetaMessage) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of TextResult objects if the response type is 'raw' and input was an iterable, a TextResult object otherwise. A list of GenerateContentResponse objects if the response type is 'raw' and input was an iterable, a GenerateContentResponse object otherwise. A list of ChatCompletion objects if the response type is 'raw' and input was an iterable, a ChatCompletion object otherwise. A list of any objects if the response type is 'raw' and input was an iterable, an any object otherwise. A list of AnthropicMessage objects if the response type is 'raw' and input was an iterable, an AnthropicMessage object otherwise. A list of AnthropicToolsBetaMessage objects if the response type is 'raw' and input was an iterable, an AnthropicToolsBetaMessage object otherwise.

        """

        assert service in ["deepl", "openai", "gemini", "google translate", "anthropic", "azure"], InvalidAPITypeException("Invalid service specified. Must be 'deepl', 'openai', 'gemini', 'google translate', 'anthropic' or 'azure'.")

        if(service == "deepl"):
            return EasyTL.deepl_translate(text, **kwargs)

        elif(service == "openai"):
            return EasyTL.openai_translate(text, **kwargs)

        elif(service == "gemini"):
           return EasyTL.gemini_translate(text, **kwargs)
        
        elif(service == "google translate"):
            return EasyTL.googletl_translate(text, **kwargs)
        
        elif(service == "anthropic"):
            return EasyTL.anthropic_translate(text, **kwargs)
        
        elif(service == "azure"):
            return EasyTL.azure_translate(text, **kwargs)
        
##-------------------start-of-translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    async def translate_async(text:str | typing.Iterable[str],
                              service:typing.Optional[typing.Literal["deepl", "openai", "gemini", "google translate", "anthropic", "azure"]] = "deepl",
                              **kwargs) -> typing.Union[typing.List[str], str, 
                                                        typing.List[TextResult], TextResult,  
                                                        typing.List[ChatCompletion], ChatCompletion,
                                                        typing.List[AsyncGenerateContentResponse], AsyncGenerateContentResponse,
                                                        typing.List[typing.Any], typing.Any,
                                                        typing.List[AnthropicMessage], AnthropicMessage]:

        
        """

        Asynchronous version of translate().
        
        Translates the given text to the target language using the specified service.
        This function assumes that the API key has already been set.
        translate_async() will generally be faster for iterables. Order is preserved.

        Please see the documentation for the specific translation function for the service you want to use.

        DeepL: deepl_translate_async()
        OpenAI: openai_translate_async() 
        Gemini: gemini_translate_async() 
        Google Translate: googletl_translate_async()
        Anthropic: anthropic_translate_async()
        Azure: azure_translate_async()

        All functions can return a list of strings or a string, depending on the input. The response type can be specified to return the raw response instead:
        DeepL: TextResult
        OpenAI: ChatCompletion
        Gemini: AsyncGenerateContentResponse
        Google Translate: any
        Anthropic: AnthropicMessage or AnthropicToolsBetaMessage
        Azure: str

        Parameters:
        service (string) : The service to use for translation.
        text (string) : The text to translate.
        **kwargs : The keyword arguments to pass to the translation function.

        Returns:
        result (string or list - string or TextResult or list - TextResult or AsyncGenerateContentResponse or list - AsyncGenerateContentResponse or ChatCompletion or list - ChatCompletion or any or list - any or AnthropicMessage or list - AnthropicMessage or AnthropicToolsBetaMessage or list - AnthropicToolsBetaMessage) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of TextResult objects if the response type is 'raw' and input was an iterable, a TextResult object otherwise. A list of AsyncGenerateContentResponse objects if the response type is 'raw' and input was an iterable, an AsyncGenerateContentResponse object otherwise. A list of ChatCompletion objects if the response type is 'raw' and input was an iterable, a ChatCompletion object otherwise. A list of any objects if the response type is 'raw' and input was an iterable, an any object otherwise. A list of AnthropicMessage objects if the response type is 'raw' and input was an iterable, an AnthropicMessage object otherwise. A list of AnthropicToolsBetaMessage objects if the response type is 'raw' and input was an iterable, an AnthropicToolsBetaMessage object otherwise.

        """

        assert service in ["deepl", "openai", "gemini", "google translate", "anthropic", "azure"], InvalidAPITypeException("Invalid service specified. Must be 'deepl', 'openai', 'gemini', 'google translate', 'anthropic' or 'azure'.")

        if(service == "deepl"):
            return await EasyTL.deepl_translate_async(text, **kwargs)

        elif(service == "openai"):
            return await EasyTL.openai_translate_async(text, **kwargs)

        elif(service == "gemini"):
            return await EasyTL.gemini_translate_async(text, **kwargs)
        
        elif(service == "google translate"):
            return await EasyTL.googletl_translate_async(text, **kwargs)
        
        elif(service == "anthropic"):
            return await EasyTL.anthropic_translate_async(text, **kwargs)
        
        elif(service == "azure"):
            return await EasyTL.azure_translate_async(text, **kwargs)

##-------------------start-of-calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def calculate_cost(text:str | typing.Iterable[str],
                       service:typing.Optional[typing.Literal["deepl", "openai", "gemini", "google translate", "anthropic", "azure"]] = "deepl",
                       model:typing.Optional[str] = None,
                       translation_instructions:typing.Optional[str] = None
                       ) -> typing.Tuple[int, float, str]:
        
        """

        Calculates the cost of translating the given text using the specified service.

        For LLMs, the cost is based on the default model unless specified.

        Model and Translation Instructions are ignored for DeepL, Google Translate and Azure.

        For DeepL, Azure and Google Translate, the number of tokens is the number of characters in the text. The returned model is the name of the service.


        Note that Anthropic's cost estimate is pretty sketchy and can be inaccurate. Refer to the actual response object for the cost or the API panel. This is because their tokenizer is not public and we're forced to estimate.

        Parameters:
        text (string or iterable) : The text to translate.
        service (string) : The service to use for translation.
        model (string or None) : The model to use for translation. If None, the default model is used.
        translation_instructions (string or None) : The translation instructions to use.

        Returns:
        num_tokens (int) : The number of tokens/characters in the text.
        cost (float) : The cost of translating the text.
        model (string) : The model used for translation.

        """

        assert service in ["deepl", "openai", "gemini", "google translate", "anthropic", "azure"], InvalidAPITypeException("Invalid service specified. Must be 'deepl', 'openai', 'gemini', 'google translate', 'anthropic' or 'azure'.")

        if(service == "deepl"):
            return DeepLService._calculate_cost(text)
        
        elif(service == "openai"):
            return OpenAIService._calculate_cost(text, translation_instructions, model)

        elif(service == "gemini"):
            return GeminiService._calculate_cost(text, translation_instructions, model)
        
        elif(service == "google translate"):
            return GoogleTLService._calculate_cost(text)
        
        elif(service == "anthropic"):
            return AnthropicService._calculate_cost(text, translation_instructions, model)
        
        elif(service == "azure"):
            return AzureService._calculate_cost(text)