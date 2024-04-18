## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio

## third-party libraries
from .classes import Language, SplitSentences, Formality, GlossaryInfo

## custom modules
from .deepl_service import DeepLService
from .gemini_service import GeminiService
from .openai_service import OpenAIService

from. classes import ModelTranslationMessage, SystemTranslationMessage, TextResult, GenerateContentResponse, AsyncGenerateContentResponse, ChatCompletion
from .exceptions import DeepLException, GoogleAPIError,OpenAIError, EasyTLException

from .util import _convert_to_correct_type, _validate_easytl_translation_settings, _is_iterable_of_strings

class EasyTL:

    """
    
    EasyTL global client, used to interact with Translation APIs.

    Use set_api_key() to set the API key for the specified API type.

    Use test_api_key_validity() to test the validity of the API key for the specified API type. (Optional) Will be done automatically when calling translation functions.

    Use translate() to translate text using the specified service with it's appropriate kwargs. Or specify the service by calling the specific translation function. (e.g. openai_translate())

    Use calculate_cost() to calculate the cost of translating text using the specified service. (Optional)

    """

##-------------------start-of-set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def set_api_key(api_type:typing.Literal["deepl", "gemini", "openai"], api_key:str) -> None:
        
        """

        Sets the API key for the specified API type.

        Parameters:
        api_type (literal["deepl", "gemini", "openai"]) : The API type to set the key for.
        api_key (string) : The API key to set.

        """

        assert api_type in ["deepl", "gemini", "openai"], ValueError("Invalid API type specified.")

        if(api_type == "deepl"):
            DeepLService._set_api_key(api_key)

        elif(api_type == "gemini"):
            GeminiService._set_api_key(api_key)
            
        elif(api_type == "openai"):
            OpenAIService._set_api_key(api_key)

##-------------------start-of-test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
    @staticmethod
    def test_api_key_validity(api_type:typing.Literal["deepl", "gemini", "openai"]) -> typing.Tuple[bool, typing.Optional[Exception]]:
        
        """

        Tests the validity of the API key for the specified API type.

        Parameters:
        api_type (literal["deepl", "gemini", "openai"]) : The API type to test the key for.

        Returns:
        (bool) : Whether the API key is valid.
        (Exception) : The exception that was raised, if any. None otherwise.

        """

        if(api_type == "deepl"):
            _is_valid, _e = DeepLService._test_api_key_validity()

            if(_is_valid == False):

                ## make sure issue is due to DeepL and not the fault of easytl, cause it needs to be raised if it is
                assert isinstance(_e, DeepLException), _e

                return False, _e
            
        if(api_type == "gemini"):
            
            _is_valid, _e = GeminiService._test_api_key_validity()

            if(_is_valid == False):

                ## make sure issue is due to Gemini and not the fault of easytl, cause it needs to be raised if it is
                assert isinstance(_e, GoogleAPIError), _e

                return False, _e
        
        if(api_type == "openai"):
            
            _is_valid, _e = OpenAIService._test_api_key_validity()

            if(_is_valid == False):

                ## make sure issue is due to OpenAI and not the fault of easytl, cause it needs to be raised if it is
                assert isinstance(_e, OpenAIError), _e

                return False, _e
            
        assert api_type in ["deepl", "gemini", "openai"], ValueError("Invalid API type specified.")

        return True, None
        
##-------------------start-of-deepl_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def deepl_translate(text:typing.Union[str, typing.Iterable[str]],
                        target_lang:str | Language = "EN-US",
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
                        logging_directory:str | None = None,
                        response_type:typing.Literal["text", "raw"] | None = "text",
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

        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string or Language) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a DeepL translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        response_type (literal["text", "raw"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a TextResult object.
        source_lang (string or Language or None) : The source language to translate from.
        context (string or None) : Additional information for the translator to be considered when translating. Not translated itself.
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
        translation (list - string or string) : The translation result. A list of strings if the input was an iterable, a string otherwise.

        """

        assert response_type in ["text", "raw"], ValueError("Invalid response type specified. Must be 'text' or 'raw'.")

        EasyTL.test_api_key_validity("deepl")

        DeepLService._set_decorator(decorator)

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
                                        semaphore=None,
                                        logging_directory=logging_directory)
            
        if(isinstance(text, str)):
            result = DeepLService._translate_text(text)
        
            assert not isinstance(result, list), "Unexpected error occurred. Please try again."

            if(response_type == "text"):
                return result.text
            
            else:
                return result
        
        elif(_is_iterable_of_strings(text)):

            results = [DeepLService._translate_text(t) for t in text]

            assert isinstance(results, list), "Unexpected error occurred. Please try again."

            if(all(isinstance(_r, TextResult) for _r in results)):
                if(response_type == "text"):
                    return [_r.text for _r in results] # type: ignore
                
                else:
                    return results # type: ignore
        
        raise ValueError("text must be a string or an iterable of strings.")
        
##-------------------start-of-deepl_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def deepl_translate_async(text:typing.Union[str, typing.Iterable[str]],
                            target_lang:str | Language = "EN-US",
                            override_previous_settings:bool = True,
                            decorator:typing.Callable | None = None,
                            logging_directory:str | None = None,
                            semaphore:int | None = None,
                            response_type:typing.Literal["text", "raw"] | None = "text",
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
        
        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string or Language) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a DeepL translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        semaphore (int) : The number of concurrent requests to make. Default is 30.
        response_type (literal["text", "raw"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a TextResult object.
        source_lang (string or Language or None) : The source language to translate from.
        context (string or None) : Additional information for the translator to be considered when translating. Not translated itself.
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
        translation (list - string or string) : The translation result. A list of strings if the input was an iterable, a string otherwise.

        """

        assert response_type in ["text", "raw"], ValueError("Invalid response type specified. Must be 'text' or 'raw'.")

        EasyTL.test_api_key_validity("deepl")

        DeepLService._set_decorator(decorator)

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
                                        semaphore=semaphore,
                                        logging_directory=logging_directory)
            
        if(isinstance(text, str)):
            _result = await DeepLService._async_translate_text(text)

            assert not isinstance(_result, list), "Unexpected error occurred. Please try again."

            if(response_type == "text"):
                return _result.text
            
            else:
                return _result
            
        elif(_is_iterable_of_strings(text)):
            _tasks = [DeepLService._async_translate_text(t) for t in text]
            _results = await asyncio.gather(*_tasks)
            
            assert isinstance(_results, list), "Unexpected error occurred. Please try again."

            if(all(isinstance(_r, TextResult) for _r in _results)):
                if(response_type == "text"):
                    return [_r.text for _r in _results] # type: ignore
                
                else:
                    return _results # type: ignore
            
        raise ValueError("text must be a string or an iterable of strings.")
            
##-------------------start-of-gemini_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def gemini_translate(text:typing.Union[str, typing.Iterable[str]],
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
                        logging_directory:str | None = None,
                        response_type:typing.Literal["text", "raw", "json"] | None = "text",
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

        It is not known whether Gemini has backoff retrying implemented. (Their API is a mess.)
        
        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Gemini translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        response_type (literal["text", "raw", "json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a GenerateContentResponse object, 'json' returns a json-parseable string.
        translation_instructions (string or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output in the instructions. The default system message will ask for json if the response type is json.
        model (string) : The model to use. 
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        top_k (int) : The top k tokens to consider. Generally, alter this or temperature or top_p, not all three.
        stop_sequences (list or None) : The sequences to stop at.
        max_output_tokens (int or None) : The maximum number of tokens to output.

        Returns:
        translation (list - string or string) : The translation result. A list of strings if the input was an iterable, a string otherwise.

        """

        assert response_type in ["text", "raw", "json"], ValueError("Invalid response type specified. Must be 'text', 'raw' or 'json'.")

        _settings = {
        "gemini_model": "",
        "gemini_temperature": "",
        "gemini_top_p": "",
        "gemini_top_k": "",
        "gemini_stop_sequences": "",
        "gemini_max_output_tokens": ""
        }

        _non_gemini_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "logging_directory", "response_type"]
        _custom_validation_params = ["gemini_stop_sequences"]

        assert stop_sequences is None or isinstance(stop_sequences, str) or (hasattr(stop_sequences, '__iter__') and all(isinstance(i, str) for i in stop_sequences)), "text must be a string or an iterable of strings."

        for _key in _settings.keys():
            param_name = _key.replace("gemini_", "")
            if(param_name in locals() and _key not in _non_gemini_params and _key not in _custom_validation_params):
                _settings[_key] = _convert_to_correct_type(_key, locals()[param_name])

        _validate_easytl_translation_settings(_settings, "gemini")

        EasyTL.test_api_key_validity("gemini")

        GeminiService._set_decorator(decorator)

        GeminiService._set_json_mode(True if response_type == "json" else False)

        translation_instructions = translation_instructions or GeminiService._system_message

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
                                          semaphore=None,
                                          logging_directory=logging_directory)

        if(isinstance(text, str)):
            _result = GeminiService._translate_text(text)
            
            if(response_type == "text" or response_type == "json"):
                return _result.text
            
            else:
                return _result

        elif(_is_iterable_of_strings(text)):
            
            _results = [GeminiService._translate_text(t) for t in text]

            if(response_type == "text" or response_type == "json"):
                return [_r.text for _r in _results]
            
            else:
                return _results
            
        raise ValueError("text must be a string or an iterable of strings.")
        
##-------------------start-of-gemini_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    async def gemini_translate_async(text:typing.Union[str, typing.Iterable[str]],
                                    override_previous_settings:bool = True,
                                    decorator:typing.Callable | None = None,
                                    logging_directory:str | None = None,
                                    semaphore:int | None = None,
                                    response_type:typing.Literal["text", "raw", "json"] | None = "text",
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

        It is not known whether Gemini has backoff retrying implemented. (Their API is a mess.)
        
        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to a Gemini translation function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        semaphore (int) : The number of concurrent requests to make. Default is 15 for 1.0 and 2 for 1.5 gemini models.
        response_type (literal["text", "raw", "json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a AsyncGenerateContentResponse object, 'json' returns a json-parseable string.
        translation_instructions (string or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output in the instructions. The default system message will ask for json if the response type is json.
        model (string) : The model to use.
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        top_k (int) : The top k tokens to consider. Generally, alter this or temperature or top_p, not all three.
        stop_sequences (list or None) : The sequences to stop at.
        max_output_tokens (int or None) : The maximum number of tokens to output.

        Returns:
        translation (list - string or string) : The translation result. A list of strings if the input was an iterable, a string otherwise.

        """

        _settings = {
        "gemini_model": "",
        "gemini_temperature": "",
        "gemini_top_p": "",
        "gemini_top_k": "",
        "gemini_stop_sequences": "",
        "gemini_max_output_tokens": ""
        }

        _non_gemini_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "logging_directory", "semaphore", "response_type"]
        _custom_validation_params = ["gemini_stop_sequences"]

        assert stop_sequences is None or isinstance(stop_sequences, str) or (hasattr(stop_sequences, '__iter__') and all(isinstance(i, str) for i in stop_sequences)), "stop_sequences must be a string or an iterable of strings."

        for _key in _settings.keys():
            param_name = _key.replace("gemini_", "")
            if(param_name in locals() and _key not in _non_gemini_params and _key not in _custom_validation_params):
                _settings[_key] = _convert_to_correct_type(_key, locals()[param_name])

        _validate_easytl_translation_settings(_settings, "gemini")

        EasyTL.test_api_key_validity("gemini")

        GeminiService._set_decorator(decorator)

        GeminiService._set_json_mode(True if response_type == "json" else False)

        translation_instructions = translation_instructions or GeminiService._system_message

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
                                          semaphore=semaphore,
                                          logging_directory=logging_directory)
            
        if(isinstance(text, str)):
            _result = await GeminiService._translate_text_async(text)

            if(response_type == "text" or response_type == "json"):
                return _result.text
            
            else:
                return _result
            
        elif(_is_iterable_of_strings(text)):
            _tasks = [GeminiService._translate_text_async(_t) for _t in text]

            _results = await asyncio.gather(*_tasks)

            if(response_type == "text" or response_type == "json"):
                return [_r.text for _r in _results]
            
            else:
                return _results 

        raise ValueError("text must be a string or an iterable of strings.")
            
##-------------------start-of-openai_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def openai_translate(text:typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]],
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
                        logging_directory:str | None = None,
                        response_type:typing.Literal["text", "raw", "json"] | None = "text",
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

        Translates the given text to the target language using OpenAI.

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
        translation_instructions (string or SystemTranslationMessage or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output in the instructions. The default system message will ask for json if the response type is json.
        model (string) : The model to use.
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        stop (list or None) : The sequences to stop at.
        max_tokens (int or None) : The maximum number of tokens to output.
        presence_penalty (float) : The presence penalty to use.
        frequency_penalty (float) : The frequency penalty to use.

        Returns:
        translation (list - string or string) : The translation result. A list of strings if the input was an iterable, a string otherwise.

        """

        _settings = {
        "openai_model": "",
        "openai_temperature": "",
        "openai_top_p": "",
        "openai_stop": "",
        "openai_max_tokens": "",
        "openai_presence_penalty": "",
        "openai_frequency_penalty": ""
        }

        _non_openai_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "logging_directory", "response_type"]
        _custom_validation_params = ["openai_stop"]
    
        assert stop is None or isinstance(stop, str) or (hasattr(stop, '__iter__') and all(isinstance(i, str) for i in stop)), "stop must be a string or an iterable of strings."

        for _key in _settings.keys():
            param_name = _key.replace("openai_", "")
            if(param_name in locals() and _key not in _non_openai_params and _key not in _custom_validation_params):
                _settings[_key] = _convert_to_correct_type(_key, locals()[param_name])

        EasyTL.test_api_key_validity("openai")

        _validate_easytl_translation_settings(_settings, "openai")

        OpenAIService._set_decorator(decorator)

        OpenAIService._set_json_mode(True if response_type == "json" else False)

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
                                        semaphore=None,
                                        logging_directory=logging_directory)
            
        translation_batches = OpenAIService._build_translation_batches(text, translation_instructions)
        
        translations = []

        for _text, _translation_instructions in translation_batches:

            _result = OpenAIService._translate_text(_translation_instructions, _text)

            if(response_type == "text" or response_type == "json"):
                translation = _result.choices[0].message.content

            else:
                translation = _result
            
            translations.append(translation)
        
        ## If originally a single text was provided, return a single translation instead of a list
        return translations if isinstance(text, typing.Iterable) and not isinstance(text, str) else translations[0]
        
##-------------------start-of-openai_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def openai_translate_async(text:typing.Union[str, typing.Iterable[str], ModelTranslationMessage, typing.Iterable[ModelTranslationMessage]],
                                    override_previous_settings:bool = True,
                                    decorator:typing.Callable | None = None,
                                    semaphore:int | None = None,
                                    logging_directory:str | None = None,
                                    response_type:typing.Literal["text", "raw", "json"] | None = "text",
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
        semaphore (int) : The number of concurrent requests to make. Default is 5.
        logging_directory (string or None) : The directory to log to. If None, no logging is done. This'll append the text result and some function information to a file in the specified directory. File is created if it doesn't exist.
        response_type (literal["text", "raw", "json"]) : The type of response to return. 'text' returns the translated text, 'raw' returns the raw response, a ChatCompletion object, 'json' returns a json-parseable string.
        translation_instructions (string or SystemTranslationMessage or None) : The translation instructions to use. If None, the default system message is used. If you plan on using the json response type, you must specify that you want a json output in the instructions. The default system message will ask for json if the response type is json.
        model (string) : The model to use.
        temperature (float) : The temperature to use. The higher the temperature, the more creative the output. Lower temperatures are typically better for translation.
        top_p (float) : The nucleus sampling probability. The higher the value, the more words are considered for the next token. Generally, alter this or temperature, not both.
        stop (list or None) : The sequences to stop at.
        max_tokens (int or None) : The maximum number of tokens to output.
        presence_penalty (float) : The presence penalty to use.
        frequency_penalty (float) : The frequency penalty to use.

        Returns:
        translation (list - string or string) : The translation result. A list of strings if the input was an iterable, a string otherwise.

        """

        _settings = {
        "openai_model": "",
        "openai_temperature": "",
        "openai_top_p": "",
        "openai_stop": "",
        "openai_max_tokens": "",
        "openai_presence_penalty": "",
        "openai_frequency_penalty": ""
        }

        _non_openai_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "logging_directory", "semaphore", "response_type"]
        _custom_validation_params = ["openai_stop"]

        assert stop is None or isinstance(stop, str) or (hasattr(stop, '__iter__') and all(isinstance(i, str) for i in stop)), "stop must be a string or an iterable of strings."

        for _key in _settings.keys():
            param_name = _key.replace("openai_", "")
            if(param_name in locals() and _key not in _non_openai_params and _key not in _custom_validation_params):
                _settings[_key] = _convert_to_correct_type(_key, locals()[param_name])

        _validate_easytl_translation_settings(_settings, "openai")

        EasyTL.test_api_key_validity("openai")

        OpenAIService._set_decorator(decorator)

        OpenAIService._set_json_mode(True if response_type == "json" else False)

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
                                        semaphore=semaphore,
                                        logging_directory=logging_directory)

        _translation_batches = OpenAIService._build_translation_batches(text, translation_instructions)

        _translation_tasks = []

        for _text, _translation_instructions in _translation_batches:
            _task = OpenAIService._translate_text_async(_translation_instructions, _text)
            _translation_tasks.append(_task)

        _results = await asyncio.gather(*_translation_tasks)

        _results:typing.List[ChatCompletion] = _results
        
        if(response_type == "text" or response_type == "json"):
            _translations = [result.choices[0].message.content for result in _results if result.choices[0].message.content is not None]

        elif(response_type == "raw"):
            _translations = _results

        # If the original input was a single text (not an iterable of texts), return a single translation instead of a list
        return _translations if isinstance(text, typing.Iterable) and not isinstance(text, str) else _translations[0]
    
##-------------------start-of-translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def translate(text:str | typing.Iterable[str],
                  service:typing.Optional[typing.Literal["deepl", "openai", "gemini"]] = "deepl", 
                  **kwargs) -> typing.Union[typing.List[str], str, 
                                            typing.List[TextResult], TextResult, 
                                            typing.List[GenerateContentResponse], GenerateContentResponse, 
                                            typing.List[ChatCompletion], ChatCompletion]:
        
        """

        Translates the given text to the target language using the specified service.

        Please see the documentation for the specific translation function for the service you want to use.

        DeepL: deepl_translate()
        OpenAI: openai_translate()
        Gemini: gemini_translate()

        Parameters:
        service (string) : The service to use for translation.
        text (string) : The text to translate.
        **kwargs : The keyword arguments to pass to the translation function.

        Returns:
        translation (TextResult or list - TextResult) : The translation result.

        """

        if(service == "deepl"):
            return EasyTL.deepl_translate(text, **kwargs)

        elif(service == "openai"):
            return EasyTL.openai_translate(text, **kwargs)

        elif(service == "gemini"):
           return EasyTL.gemini_translate(text, **kwargs)
        
        else:
            raise ValueError("Invalid service specified.")
        
##-------------------start-of-translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    async def translate_async(text:str | typing.Iterable[str],
                              service:typing.Optional[typing.Literal["deepl", "openai", "gemini"]] = "deepl", 
                              **kwargs) -> typing.Union[typing.List[str], str, 
                                                        typing.List[TextResult], TextResult, 
                                                        typing.List[GenerateContentResponse], GenerateContentResponse, 
                                                        typing.List[ChatCompletion], ChatCompletion,
                                                        AsyncGenerateContentResponse, typing.List[AsyncGenerateContentResponse]]:
        
        """

        Asynchronous version of translate().
        
        Translates the given text to the target language using the specified service.
        This function assumes that the API key has already been set.
        translate_async() will generally be faster for iterables. Order is preserved.

        Please see the documentation for the specific translation function for the service you want to use.

        DeepL: deepl_translate_async()
        OpenAI: openai_translate_async()
        Gemini: gemini_translate_async()

        Parameters:
        service (string) : The service to use for translation.
        text (string) : The text to translate.
        **kwargs : The keyword arguments to pass to the translation function.

        Returns:
        translation (TextResult or list - TextResult) : The translation result.

        """

        if(service == "deepl"):
            return await EasyTL.deepl_translate_async(text, **kwargs)

        elif(service == "openai"):
            return await EasyTL.openai_translate_async(text, **kwargs)

        elif(service == "gemini"):
            return await EasyTL.gemini_translate_async(text, **kwargs)
        
        else:
            raise ValueError("Invalid service specified.")
        
##-------------------start-of-calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def calculate_cost(text:str | typing.Iterable[str],
                       service:typing.Optional[typing.Literal["deepl", "openai", "gemini"]] = "deepl",
                       model:typing.Optional[str] = None,
                       translation_instructions:typing.Optional[str] = None
                       ) -> typing.Tuple[int, float, str]:
        
        """

        Calculates the cost of translating the given text using the specified service.

        For LLMs, the cost is based on the default model unless specified.

        Model and Translation Instructions are ignored for DeepL.

        For deepl, number of tokens is the number of characters, the returned model is always "deepl"

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

        if(service == "deepl"):
            return DeepLService._calculate_cost(text)
        
        elif(service == "openai"):
            return OpenAIService._calculate_cost(text, translation_instructions, model)

        elif(service == "gemini"):
            return GeminiService._calculate_cost(text, translation_instructions, model)
        
        else:
            raise ValueError("Invalid service specified.")