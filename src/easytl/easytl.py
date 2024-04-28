## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio

import warnings

## third-party libraries
from .classes import Language, SplitSentences, Formality, GlossaryInfo

## custom modules
from .deepl_service import DeepLService
from .gemini_service import GeminiService
from .openai_service import OpenAIService
from .googletl_service import GoogleTLService

from. classes import ModelTranslationMessage, SystemTranslationMessage, TextResult, GenerateContentResponse, AsyncGenerateContentResponse, ChatCompletion
from .exceptions import DeepLException, GoogleAPIError, OpenAIError, InvalidAPITypeException, InvalidResponseFormatException, InvalidTextInputException, EasyTLException

from .util import _validate_easytl_translation_settings, _is_iterable_of_strings, _return_curated_gemini_settings, _return_curated_openai_settings, _validate_stop_sequences

class EasyTL:

    """
    
    EasyTL global client, used to interact with Translation APIs.

    Use set_credentials() to set the credentials for the specified API type. (e.g. set_credentials("deepl", "your_api_key") or set_credentials("google translate", "path/to/your/credentials.json"))

    Use test_credentials() to test the validity of the credentials for the specified API type. (e.g. test_credentials("deepl")) (Optional) Done automatically when translating.

    Use translate() to translate text using the specified service with it's appropriate kwargs. Or specify the service by calling the specific translation function. (e.g. openai_translate())

    Use calculate_cost() to calculate the cost of translating text using the specified service. (Optional)

    See the documentation for each function for more information.

    """

##-------------------start-of-set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def set_api_key(api_type:typing.Literal["deepl", "gemini", "openai"], api_key:str) -> None:
        
        """

        Sets the API key for the specified API type.
        For Google Translate, use set_credentials() instead.

        Deprecated: This function is deprecated and will be removed in a future version.
        Use set_credentials(auth_info) instead.

        Parameters:
        api_type (literal["deepl", "gemini", "openai"]) : The API type to set the key for.
        api_key (string) : The API key to set.

        """

        warnings.warn("set_api_key is deprecated and will be removed in a future version. Use set_credentials instead.", DeprecationWarning, stacklevel=2)

        service_map = {
            "deepl": DeepLService,
            "gemini": GeminiService,
            "openai": OpenAIService
        }

        assert api_type in service_map, InvalidAPITypeException("Invalid API type specified. Supported types are 'deepl', 'gemini' and 'openai'.")

        service_map[api_type]._set_api_key(api_key)

##-------------------start-of-set_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def set_credentials(api_type:typing.Literal["deepl", "gemini", "openai", "google translate"], credentials:str) -> None:

        """

        Sets the credentials for the specified API type.

        Parameters:
        api_type (literal["deepl", "gemini", "openai", "google translate"]) : The API type to set the credentials for.
        credentials (string) : The credentials to set. This is an api key for deepl, gemini and openai. For google translate, this is a path to your json that has your service account key.

        """

        service_map = {
            "deepl": DeepLService._set_api_key,
            "gemini": GeminiService._set_api_key,
            "openai": OpenAIService._set_api_key,
            "google translate": GoogleTLService._set_credentials

        }

        assert api_type in service_map, InvalidAPITypeException("Invalid API type specified. Supported types are 'deepl', 'gemini', 'openai' and 'google translate'.")

        service_map[api_type](credentials)

##-------------------start-of-test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
    @staticmethod
    def test_api_key_validity(api_type:typing.Literal["deepl", "gemini", "openai"]) -> typing.Tuple[bool, typing.Optional[Exception]]:
        
        """

        Tests the validity of the API key for the specified API type.
        
        Deprecated: This function is deprecated and will be removed in a future version.
        Use test_credentials() instead.

        Parameters:
        api_type (literal["deepl", "gemini", "openai"]) : The API type to test the key for.

        Returns:
        (bool) : Whether the API key is valid.
        (Exception) : The exception that was raised, if any. None otherwise.

        """

        warnings.warn("test_api_key_validity is deprecated and will be removed in a future version. Use test_credentials instead.", DeprecationWarning, stacklevel=2)
        
        api_services = {
            "deepl": {"service": DeepLService, "exception": DeepLException},
            "gemini": {"service": GeminiService, "exception": GoogleAPIError},
            "openai": {"service": OpenAIService, "exception": OpenAIError}
        }

        assert api_type in api_services, InvalidAPITypeException("Invalid API type specified. Supported types are 'deepl', 'gemini' and 'openai'.")

        _is_valid, _e = api_services[api_type]["service"]._test_api_key_validity()

        if(not _is_valid):
            ## Done to make sure the exception is due to the specified API type and not the fault of EasyTL
            assert isinstance(_e, api_services[api_type]["exception"]), _e
            return False, _e

        return True, None
    
##-------------------start-of-test_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def test_credentials(api_type:typing.Literal["deepl", "gemini", "openai", "google translate"]) -> typing.Tuple[bool, typing.Optional[Exception]]:

        """

        Tests the validity of the credentials for the specified API type.

        Parameters:
        api_type (literal["deepl", "gemini", "openai", "google translate"]) : The API type to test the credentials for.

        Returns:
        (bool) : Whether the credentials are valid.
        (Exception) : The exception that was raised, if any. None otherwise.

        """

        api_services = {
            "deepl": {"service": DeepLService, "exception": DeepLException, "test_func": DeepLService._test_api_key_validity},
            "gemini": {"service": GeminiService, "exception": GoogleAPIError, "test_func": GeminiService._test_api_key_validity},
            "openai": {"service": OpenAIService, "exception": OpenAIError, "test_func": OpenAIService._test_api_key_validity},
            "google translate": {"service": GoogleTLService, "exception": GoogleAPIError, "test_func": GoogleTLService._test_credentials}
        }

        assert api_type in api_services, InvalidAPITypeException("Invalid API type specified. Supported types are 'deepl', 'gemini', 'openai' and 'google translate'.")

        _is_valid, _e = api_services[api_type]["test_func"]()

        if(not _is_valid):
            ## Done to make sure the exception is due to the specified API type and not the fault of EasyTL
            assert isinstance(_e, api_services[api_type]["exception"]), _e
            return False, _e

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
        
        """

        Translates the given text to the target language using Google Translate.

        This function assumes that the credentials have already been set.

        It is unknown whether Google Translate has backoff retrying implemented. Assume it does not exist.

        Due to how Google Translate's API works, the translation delay and semaphore are not as important as they are for other services. As they process iterables directly.

        Google Translate v2 API is poorly documented and type hints are near non-existent. typing.Any return types are used for the raw response type.

        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Google Translate function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
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
                                            logging_directory=logging_directory, 
                                            semaphore=None, 
                                            rate_limit_delay=translation_delay)
            
        if(isinstance(text, str)):
            result = GoogleTLService._translate_text(text)
        
            assert not isinstance(result, list), EasyTLException("Malformed response received. Please try again.")

            result = result if response_type == "raw" else result["translatedText"]
        
        elif(_is_iterable_of_strings(text)):

            results = [GoogleTLService._translate_text(t) for t in text]

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
                                       semaphore:int | None = None,
                                       translation_delay:float | None = None,
                                       format:typing.Literal["text", "html"] = "text",
                                       source_lang:str | None = None) -> typing.Union[typing.List[str], str, typing.List[typing.Any], typing.Any]:
        
        """

        Asynchronous version of googletl_translate().

        Translates the given text to the target language using Google Translate.
        Will generally be faster for iterables. Order is preserved.

        This function assumes that the credentials have already been set.

        It is unknown whether Google Translate has backoff retrying implemented. Assume it does not exist.

        Due to how Google Translate's API works, the translation delay and semaphore are not as important as they are for other services. As they process iterables directly.

        Google Translate v2 API is poorly documented and type hints are near non-existent. typing.Any return types are used for the raw response type.

        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Google Translate function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
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
                                            logging_directory=logging_directory, 
                                            semaphore=semaphore, 
                                            rate_limit_delay=translation_delay)
            
        if(isinstance(text, str)):
            _result = await GoogleTLService._translate_text_async(text)

            assert not isinstance(_result, list), EasyTLException("Malformed response received. Please try again.")

            result = _result if response_type == "raw" else _result["translatedText"]
            
        elif(_is_iterable_of_strings(text)):
            _tasks = [GoogleTLService._translate_text_async(t) for t in text]
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

        DeepL has backoff retrying implemented by default.

        Due to how DeepL's API works, the translation delay and semaphore are not as important as they are for other services. As they process iterables directly.

        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string or Language) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a DeepL translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
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
                                        logging_directory=logging_directory,
                                        semaphore=None,
                                        rate_limit_delay=translation_delay)
            
        if(isinstance(text, str)):
            result = DeepLService._translate_text(text)
        
            assert not isinstance(result, list), EasyTLException("Malformed response received. Please try again.")

            result = result if response_type == "raw" else result.text
        
        elif(_is_iterable_of_strings(text)):

            results = [DeepLService._translate_text(t) for t in text]

            assert isinstance(results, list), EasyTLException("Malformed response received. Please try again.")

            result = [_r.text for _r in results] if response_type == "text" else results # type: ignore    
            
        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result
        
##-------------------start-of-deepl_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def deepl_translate_async(text:typing.Union[str, typing.Iterable[str]],
                            target_lang:str | Language = "EN-US",
                            override_previous_settings:bool = True,
                            decorator:typing.Callable | None = None,
                            logging_directory:str | None = None,
                            response_type:typing.Literal["text", "raw"] | None = "text",
                            semaphore:int | None = None,
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

        DeepL has backoff retrying implemented by default.

        Due to how DeepL's API works, the translation delay and semaphore are not as important as they are for other services. As they process iterables directly.
        
        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string or Language) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a DeepL translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        response_type (literal["text", "raw"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a TextResult object.
        semaphore (int) : The number of concurrent requests to make. Default is 30.
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
                                        logging_directory=logging_directory,
                                        semaphore=semaphore,
                                        rate_limit_delay=translation_delay)
        if(isinstance(text, str)):
            _result = await DeepLService._translate_text_async(text)

            assert not isinstance(_result, list), EasyTLException("Malformed response received. Please try again.")

            result = _result if response_type == "raw" else _result.text
            
        elif(_is_iterable_of_strings(text)):
            _tasks = [DeepLService._translate_text_async(t) for t in text]
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
                        response_type:typing.Literal["text", "raw", "json"] | None = "text",
                        translation_delay:float | None = None,
                        translation_instructions:str | None = None,
                        model:str="gemini-pro",
                        temperature:float=0.5,
                        top_p:float=0.9,
                        top_k:int=40,
                        stop_sequences:typing.List[str] | None=None,
                        max_output_tokens:int | None=None) -> typing.Union[typing.List[str], str, GenerateContentResponse, typing.List[GenerateContentResponse]]:
        
        """

        Translates the given text using Gemini.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        It is not known whether Gemini has backoff retrying implemented. Assume it does not exist. 
        
        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Gemini translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        response_type (literal["text", "raw", "json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a GenerateContentResponse object, 'json' returns a json-parseable string.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        model (string) : The model to use. 
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        top_k (int) : The top k tokens to consider. Generally, alter this or temperature or top_p, not all three.
        stop_sequences (list or None) : The sequences to stop at.
        max_output_tokens (int or None) : The maximum number of tokens to output.

        Returns:
        result (string or list - string or GenerateContentResponse or list - GenerateContentResponse) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of GenerateContentResponse objects if the response type is 'raw' and input was an iterable, a GenerateContentResponse object otherwise.

        """

        assert response_type in ["text", "raw", "json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw' or 'json'.")

        _settings = _return_curated_gemini_settings(locals())

        _validate_easytl_translation_settings(_settings, "gemini")

        _validate_stop_sequences(stop_sequences)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("gemini")

        json_mode = True if response_type == "json" else False

        if(override_previous_settings == True):
            GeminiService._set_attributes(model=model,
                                          system_message=translation_instructions,
                                          temperature=temperature,
                                          top_p=top_p,
                                          top_k=top_k,
                                          candidate_count=1,
                                          stream=False,
                                          stop_sequences=stop_sequences,
                                          max_output_tokens=max_output_tokens,
                                          decorator=decorator,
                                          logging_directory=logging_directory,
                                          semaphore=None,
                                          rate_limit_delay=translation_delay,
                                          json_mode=json_mode)
            
            ## Done afterwards, cause default translation instructions can change based on set_attributes()       
            GeminiService._system_message = translation_instructions or GeminiService._default_translation_instructions
        
        if(isinstance(text, str)):
            _result = GeminiService._translate_text(text)
            
            assert not isinstance(_result, list) and hasattr(_result, "text"), EasyTLException("Malformed response received. Please try again.")
            
            result = _result if response_type == "raw" else _result.text

        elif(_is_iterable_of_strings(text)):
            
            _results = [GeminiService._translate_text(t) for t in text]

            assert isinstance(_results, list) and all([hasattr(_r, "text") for _r in _results]), EasyTLException("Malformed response received. Please try again.")

            result = [_r.text for _r in _results] if response_type == "text" else _results # type: ignore
            
        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result

##-------------------start-of-gemini_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    async def gemini_translate_async(text:typing.Union[str, typing.Iterable[str]],
                                    override_previous_settings:bool = True,
                                    decorator:typing.Callable | None = None,
                                    logging_directory:str | None = None,
                                    response_type:typing.Literal["text", "raw", "json"] | None = "text",
                                    semaphore:int | None = None,
                                    translation_delay:float | None = None,
                                    translation_instructions:str | None = None,
                                    model:str="gemini-pro",
                                    temperature:float=0.5,
                                    top_p:float=0.9,
                                    top_k:int=40,
                                    stop_sequences:typing.List[str] | None=None,
                                    max_output_tokens:int | None=None) -> typing.Union[typing.List[str], str, AsyncGenerateContentResponse, typing.List[AsyncGenerateContentResponse]]:
        
        """

        Asynchronous version of gemini_translate().
        Will generally be faster for iterables. Order is preserved.

        Translates the given text using Gemini.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        It is not known whether Gemini has backoff retrying implemented. Assume it does not exist.
        
        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Gemini translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        response_type (literal["text", "raw", "json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a AsyncGenerateContentResponse object, 'json' returns a json-parseable string.
        semaphore (int) : The number of concurrent requests to make. Default is 15 for 1.0 and 2 for 1.5 gemini models. For Gemini, it is recommend to use translation_delay along with the semaphore to prevent rate limiting.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        model (string) : The model to use.
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        top_k (int) : The top k tokens to consider. Generally, alter this or temperature or top_p, not all three.
        stop_sequences (list or None) : The sequences to stop at.
        max_output_tokens (int or None) : The maximum number of tokens to output.

        Returns:
        result (string or list - string or AsyncGenerateContentResponse or list - AsyncGenerateContentResponse) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of AsyncGenerateContentResponse objects if the response type is 'raw' and input was an iterable, a AsyncGenerateContentResponse object otherwise.

        """

        assert response_type in ["text", "raw", "json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw' or 'json'.")

        _settings = _return_curated_gemini_settings(locals())

        _validate_easytl_translation_settings(_settings, "gemini")

        _validate_stop_sequences(stop_sequences)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("gemini")

        json_mode = True if response_type == "json" else False

        if(override_previous_settings == True):
            GeminiService._set_attributes(model=model,
                                          system_message=translation_instructions,
                                          temperature=temperature,
                                          top_p=top_p,
                                          top_k=top_k,
                                          candidate_count=1,
                                          stream=False,
                                          stop_sequences=stop_sequences,
                                          max_output_tokens=max_output_tokens,
                                          decorator=decorator,
                                          logging_directory=logging_directory,
                                          semaphore=semaphore,
                                          rate_limit_delay=translation_delay,
                                          json_mode=json_mode)
            
            ## Done afterwards, cause default translation instructions can change based on set_attributes()
            GeminiService._system_message = translation_instructions or GeminiService._default_translation_instructions
            
        if(isinstance(text, str)):
            _result = await GeminiService._translate_text_async(text)

            result = _result if response_type == "raw" else _result.text
            
        elif(_is_iterable_of_strings(text)):
            _tasks = [GeminiService._translate_text_async(_t) for _t in text]
            _results = await asyncio.gather(*_tasks)

            result = [_r.text for _r in _results] if response_type == "text" else _results # type: ignore

        else:
            raise InvalidTextInputException("text must be a string or an iterable of strings.")
        
        return result
            
##-------------------start-of-openai_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def openai_translate(text:typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]],
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
                        logging_directory:str | None = None,
                        response_type:typing.Literal["text", "raw", "json"] | None = "text",
                        translation_delay:float | None = None,
                        translation_instructions:str | SystemTranslationMessage | None = None,
                        model:str="gpt-4",
                        temperature:float=0.3,
                        top_p:float=1.0,
                        stop:typing.List[str] | None=None,
                        max_tokens:int | None=None,
                        presence_penalty:float=0.0,
                        frequency_penalty:float=0.0
                        ) -> typing.Union[typing.List[str], str, typing.List[ChatCompletion], ChatCompletion]:
        
        """

        Translates the given text using OpenAI.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        OpenAI has it's backoff retrying disabled by EasyTL, in favor of the user implementing their own retrying mechanism via the decorator.

        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to an OpenAI translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        response_type (literal["text", "raw", "json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a ChatCompletion object, 'json' returns a json-parseable string.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or SystemTranslationMessage or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        stop (list or None) : The sequences to stop at.
        max_tokens (int or None) : The maximum number of tokens to output.
        presence_penalty (float) : The presence penalty to use.
        frequency_penalty (float) : The frequency penalty to use.

        Returns:
        result (string or list - string or ChatCompletion or list - ChatCompletion) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of ChatCompletion objects if the response type is 'raw' and input was an iterable, a ChatCompletion object otherwise.

        """

        assert response_type in ["text", "raw", "json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw' or 'json'.")

        _settings = _return_curated_openai_settings(locals())

        _validate_easytl_translation_settings(_settings, "openai")

        _validate_stop_sequences(stop)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("openai")

        json_mode = True if response_type == "json" else False
        
        if(override_previous_settings == True):
            OpenAIService._set_attributes(model=model,
                                        temperature=temperature,
                                        logit_bias=None,
                                        top_p=top_p,
                                        n=1,
                                        stop=stop,
                                        max_tokens=max_tokens,
                                        presence_penalty=presence_penalty,
                                        frequency_penalty=frequency_penalty,
                                        decorator=decorator,
                                        logging_directory=logging_directory,
                                        semaphore=None,
                                        rate_limit_delay=translation_delay,
                                        json_mode=json_mode)

            ## Done afterwards, cause default translation instructions can change based on set_attributes()
            translation_instructions = translation_instructions or OpenAIService._default_translation_instructions
        
        else:
            translation_instructions = OpenAIService._system_message

        assert isinstance(text, str) or _is_iterable_of_strings(text) or isinstance(text, ModelTranslationMessage) or _is_iterable_of_strings(text), InvalidTextInputException("text must be a string, an iterable of strings, a ModelTranslationMessage or an iterable of ModelTranslationMessages.")

        translation_batches = OpenAIService._build_translation_batches(text, translation_instructions)
        
        translations = []
        
        for _text, _translation_instructions in translation_batches:

            _result = OpenAIService._translate_text(_translation_instructions, _text)

            assert not isinstance(_result, list) and hasattr(_result, "choices"), EasyTLException("Malformed response received. Please try again.")

            translation = _result if response_type == "raw" else _result.choices[0].message.content
            
            translations.append(translation)
        
        ## If originally a single text was provided, return a single translation instead of a list
        result = translations if isinstance(text, typing.Iterable) and not isinstance(text, str) else translations[0]
        
        return result
    
##-------------------start-of-openai_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def openai_translate_async(text:typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]],
                                    override_previous_settings:bool = True,
                                    decorator:typing.Callable | None = None,
                                    logging_directory:str | None = None,
                                    response_type:typing.Literal["text", "raw", "json"] | None = "text",
                                    semaphore:int | None = None,
                                    translation_delay:float | None = None,
                                    translation_instructions:str | SystemTranslationMessage | None = None,
                                    model:str="gpt-4",
                                    temperature:float=0.3,
                                    top_p:float=1.0,
                                    stop:typing.List[str] | None=None,
                                    max_tokens:int | None=None,
                                    presence_penalty:float=0.0,
                                    frequency_penalty:float=0.0
                                    ) -> typing.Union[typing.List[str], str, typing.List[ChatCompletion], ChatCompletion]:
        
        """

        Asynchronous version of openai_translate().
        Will generally be faster for iterables. Order is preserved.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        OpenAI has it's backoff retrying disabled by EasyTL.

        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to an OpenAI translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        response_type (literal["text", "raw", "json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a ChatCompletion object, 'json' returns a json-parseable string.
        semaphore (int) : The number of concurrent requests to make. Default is 5.
        translation_delay (float or None) : If text is an iterable, the delay between each translation. Default is none. This is more important for asynchronous translations where a semaphore alone may not be sufficient.
        translation_instructions (string or SystemTranslationMessage or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output and it's format in the instructions. The default system message will ask for a generic json if the response type is json.
        model (string) : The model to use.
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        stop (list or None) : The sequences to stop at.
        max_tokens (int or None) : The maximum number of tokens to output.
        presence_penalty (float) : The presence penalty to use.
        frequency_penalty (float) : The frequency penalty to use.

        Returns:
        result (string or list - string or ChatCompletion or list - ChatCompletion) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of ChatCompletion objects if the response type is 'raw' and input was an iterable, a ChatCompletion object otherwise.
        
        """

        assert response_type in ["text", "raw", "json"], InvalidResponseFormatException("Invalid response type specified. Must be 'text', 'raw' or 'json'.")

        _settings = _return_curated_openai_settings(locals())

        _validate_easytl_translation_settings(_settings, "openai")

        _validate_stop_sequences(stop)

        ## Should be done after validating the settings to reduce cost to the user
        EasyTL.test_credentials("openai")

        json_mode = True if response_type == "json" else False

        if(override_previous_settings == True):
            OpenAIService._set_attributes(model=model,
                                        temperature=temperature,
                                        logit_bias=None,
                                        top_p=top_p,
                                        n=1,
                                        stop=stop,
                                        max_tokens=max_tokens,
                                        presence_penalty=presence_penalty,
                                        frequency_penalty=frequency_penalty,
                                        decorator=decorator,
                                        logging_directory=logging_directory,
                                        semaphore=semaphore,
                                        rate_limit_delay=translation_delay,
                                        json_mode=json_mode)
            
            ## Done afterwards, cause default translation instructions can change based on set_attributes()
            translation_instructions = translation_instructions or OpenAIService._default_translation_instructions

        else:
            translation_instructions = OpenAIService._system_message        
            
        assert isinstance(text, str) or _is_iterable_of_strings(text) or isinstance(text, ModelTranslationMessage) or _is_iterable_of_strings(text), InvalidTextInputException("text must be a string, an iterable of strings, a ModelTranslationMessage or an iterable of ModelTranslationMessages.")

        _translation_batches = OpenAIService._build_translation_batches(text, translation_instructions)

        _translation_tasks = []

        for _text, _translation_instructions in _translation_batches:
            _task = OpenAIService._translate_text_async(_translation_instructions, _text)
            _translation_tasks.append(_task)

        _results = await asyncio.gather(*_translation_tasks)

        _results:typing.List[ChatCompletion] = _results

        assert all([hasattr(_r, "choices") for _r in _results]), EasyTLException("Malformed response received. Please try again.")

        translation = _results if response_type == "raw" else [result.choices[0].message.content for result in _results if result.choices[0].message.content is not None]

        result = translation if isinstance(text, typing.Iterable) and not isinstance(text, str) else translation[0]

        return result
    
##-------------------start-of-translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def translate(text:str | typing.Iterable[str],
                  service:typing.Optional[typing.Literal["deepl", "openai", "gemini", "google translate"]] = "deepl",
                  **kwargs) -> typing.Union[typing.List[str], str, 
                                            typing.List[TextResult], TextResult, 
                                            typing.List[ChatCompletion], ChatCompletion,
                                            typing.List[GenerateContentResponse], GenerateContentResponse, 
                                            typing.List[typing.Any], typing.Any]:
        
        """

        Translates the given text to the target language using the specified service.

        Please see the documentation for the specific translation function for the service you want to use.

        DeepL: deepl_translate() 
        OpenAI: openai_translate() 
        Gemini: gemini_translate() 
        Google Translate: googletl_translate() 

        All functions can return a list of strings or a string, depending on the input. The response type can be specified to return the raw response instead:
        DeepL: TextResult
        OpenAI: ChatCompletion
        Gemini: GenerateContentResponse
        Google Translate: any

        Parameters:
        service (string) : The service to use for translation.
        text (string) : The text to translate.
        **kwargs : The keyword arguments to pass to the translation function.

        Returns:
        result (string or list - string or TextResult or list - TextResult or ChatCompletion or list - ChatCompletion or GenerateContentResponse or list - GenerateContentResponse or any or list - any) : The translation result. A list of strings if the input was an iterable, a string otherwise. A list of TextResult objects if the response type is 'raw' and input was an iterable, a TextResult object otherwise. A list of ChatCompletion objects if the response type is 'raw' and input was an iterable, a ChatCompletion object otherwise. A list of GenerateContentResponse objects if the response type is 'raw' and input was an iterable, a GenerateContentResponse object otherwise. A list of any objects if the response type is 'raw' and input was an iterable, an any object otherwise.

        """

        assert service in ["deepl", "openai", "gemini", "google translate"], InvalidAPITypeException("Invalid service specified. Must be 'deepl', 'openai', 'gemini' or 'google translate'.")

        if(service == "deepl"):
            return EasyTL.deepl_translate(text, **kwargs)

        elif(service == "openai"):
            return EasyTL.openai_translate(text, **kwargs)

        elif(service == "gemini"):
           return EasyTL.gemini_translate(text, **kwargs)
        
        elif(service == "google translate"):
            return EasyTL.googletl_translate(text, **kwargs)
        
##-------------------start-of-translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    async def translate_async(text:str | typing.Iterable[str],
                              service:typing.Optional[typing.Literal["deepl", "openai", "gemini", "google translate"]] = "deepl",
                              **kwargs) -> typing.Union[typing.List[str], str, 
                                                        typing.List[TextResult], TextResult,  
                                                        typing.List[ChatCompletion], ChatCompletion,
                                                        typing.List[AsyncGenerateContentResponse], AsyncGenerateContentResponse,
                                                        typing.List[typing.Any], typing.Any]:

        
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

        All functions can return a list of strings or a string, depending on the input. The response type can be specified to return the raw response instead:
        DeepL: TextResult
        OpenAI: ChatCompletion
        Gemini: AsyncGenerateContentResponse
        Google Translate: any

        Parameters:
        service (string) : The service to use for translation.
        text (string) : The text to translate.
        **kwargs : The keyword arguments to pass to the translation function.

        Returns:
        result (string or list - string or TextResult or list - TextResult or AsyncGenerateContentResponse or list - AsyncGenerateContentResponse or ChatCompletion or list - ChatCompletion or any or list - any) : The translation result according to the service used.

        """

        assert service in ["deepl", "openai", "gemini", "google translate"], InvalidAPITypeException("Invalid service specified. Must be 'deepl', 'openai', 'gemini' or 'google translate'.")

        if(service == "deepl"):
            return await EasyTL.deepl_translate_async(text, **kwargs)

        elif(service == "openai"):
            return await EasyTL.openai_translate_async(text, **kwargs)

        elif(service == "gemini"):
            return await EasyTL.gemini_translate_async(text, **kwargs)
        
        elif(service == "google translate"):
            return await EasyTL.googletl_translate_async(text, **kwargs)

##-------------------start-of-calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def calculate_cost(text:str | typing.Iterable[str],
                       service:typing.Optional[typing.Literal["deepl", "openai", "gemini", "google translate"]] = "deepl",
                       model:typing.Optional[str] = None,
                       translation_instructions:typing.Optional[str] = None
                       ) -> typing.Tuple[int, float, str]:
        
        """

        Calculates the cost of translating the given text using the specified service.

        For LLMs, the cost is based on the default model unless specified.

        Model and Translation Instructions are ignored for DeepL and Google Translate.

        For deepl, number of tokens is the number of characters, the returned model is always "deepl".
        The same applies for google translate, but the model is "google translate".

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

        assert service in ["deepl", "openai", "gemini", "google translate"], InvalidAPITypeException("Invalid service specified. Must be 'deepl', 'openai', 'gemini' or 'google translate'.")

        if(service == "deepl"):
            return DeepLService._calculate_cost(text)
        
        elif(service == "openai"):
            return OpenAIService._calculate_cost(text, translation_instructions, model)

        elif(service == "gemini"):
            return GeminiService._calculate_cost(text, translation_instructions, model)
        
        elif(service == "google translate"):
            return GoogleTLService._calculate_cost(text)