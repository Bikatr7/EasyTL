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

from .exceptions import DeepLException, GoogleAPIError

from .util import convert_to_correct_type, validate_easytl_translation_settings

class EasyTL:

    """
    
    EasyTL global client, used to interact with Translation APIs.

    Use set_api_key() to set the API key for the specified API type.

    Use test_api_key_validity() to test the validity of the API key for the specified API type. (Optional) Will be done automatically when calling translation functions.

    Use translate() to translate text using the specified service. Or specify the service by calling the specific translation function. (e.g. deepl_translate())

    """

##-------------------start-of-set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def set_api_key(api_type:typing.Literal["deepl", "gemini", "openai"], api_key:str) -> None:
        
        """

        Sets the API key for the specified API type.

        Parameters:
        api_type (string) : The API type to set the key for.
        api_key (string) : The API key to set.

        """

        if(api_type == "deepl"):
            DeepLService._set_api_key(api_key)

        elif(api_type == "gemini"):
            GeminiService._set_api_key(api_key)
            
        elif(api_type == "openai"):
            OpenAIService.set_api_key(api_key)

##-------------------start-of-test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
    @staticmethod
    def test_api_key_validity(api_type:typing.Literal["deepl", "gemini", "openai"]) -> typing.Tuple[bool, typing.Optional[Exception]]:
        
        """

        Tests the validity of the API key for the specified API type.

        Parameters:
        api_type (string) : The API type to test the key for.

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
            raise NotImplementedError("OpenAI service is not yet implemented.")

        return True, None
        
        ## need to add the other services here

##-------------------start-of-deepl_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def deepl_translate(text:typing.Union[str, typing.Iterable],
                        target_lang:str | Language = "EN",
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
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
                        ignore_tags:str | typing.List[str] | None = None) -> typing.Union[typing.List[str], str]:
        
        """

        Translates the given text to the target language using DeepL.

        This function assumes that the API key has already been set.

        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string or Language) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to this function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
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

        if(decorator != None):
            DeepLService._set_decorator(decorator)

        if(override_previous_settings == True):
            DeepLService._set_attributes(target_lang, 
                                        source_lang, 
                                        context, 
                                        split_sentences,
                                        preserve_formatting, 
                                        formality, 
                                        glossary, tag_handling, outline_detection, non_splitting_tags, splitting_tags, ignore_tags)
            
        if(isinstance(text, str)):
            return DeepLService._translate_text(text).text # type: ignore
        
        else:
            return [DeepLService._translate_text(t).text for t in text] # type: ignore
        
##-------------------start-of-deepl_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def deepl_translate_async(text:typing.Union[str, typing.Iterable],
                            target_lang:str | Language = "EN",
                            override_previous_settings:bool = True,
                            decorator:typing.Callable | None = None,
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
                            ignore_tags:str | typing.List[str] | None = None) -> typing.Union[typing.List[str], str]:
        
        """

        Asynchronous version of deepl_translate().
        
        Translates the given text to the target language using DeepL.
        Will generally be faster for iterables. Order is preserved.

        This function assumes that the API key has already been set.
        
        Parameters:
        text (string or iterable) : The text to translate.
        target_lang (string or Language) : The target language to translate to.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to this function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
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

        if(decorator != None):
            DeepLService._set_decorator(decorator)

        if(override_previous_settings == True):
            DeepLService._set_attributes(target_lang, 
                                        source_lang, 
                                        context, 
                                        split_sentences,
                                        preserve_formatting, 
                                        formality, 
                                        glossary, tag_handling, outline_detection, non_splitting_tags, splitting_tags, ignore_tags)
            
        if(isinstance(text, str)):
            _result = DeepLService._async_translate_text(text)

            if(hasattr(_result, "text")):
                return _result.text # type: ignore
            
            else:
                raise Exception("Unexpected error occurred. Please try again.")
            
        elif isinstance(text, typing.Iterable):
            _tasks = [DeepLService._async_translate_text(t) for t in text]
            _results = []

            for _future in asyncio.as_completed(_tasks):
                _result = await _future
                _results.append(_result)

            if(all(hasattr(_r, "text") for _r in _results)):
                return [_r.text for _r in results]  # type: ignore
            
            else:
                raise Exception("Unexpected error occurred. Please try again.")
            
##-------------------start-of-gemini_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def gemini_translate(text:typing.Union[str, typing.Iterable],
                        override_previous_settings:bool = True,
                        decorator:typing.Callable | None = None,
                        translation_instructions:str | None = None,
                        model:str="gemini-pro",
                        temperature:float=0.5,
                        top_p:float=0.9,
                        top_k:int=40,
                        stop_sequences:typing.List[str] | None=None,
                        max_output_tokens:int | None=None) -> str | typing.List[str]:
        
        """

        Translates the given text using Gemini.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to this function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        translation_instructions (string or None) : The translation instructions to use.
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

        _non_gemini_params = ["text", "override_previous_settings", "decorator", "translation_instructions"]

        for _key, _value in locals().items():
            if(_key not in _non_gemini_params):
                _settings[_key] = convert_to_correct_type(_key, _value)

        validate_easytl_translation_settings(_settings, "gemini")

        if(decorator != None):
            GeminiService._set_decorator(decorator)

        if(override_previous_settings == True):
            GeminiService._set_attributes(model=model,
                                          temperature=temperature,
                                          top_p=top_p,
                                          top_k=top_k,
                                          candidate_count=1,
                                          stream=False,
                                          stop_sequences=stop_sequences,
                                          max_output_tokens=max_output_tokens)

        if(isinstance(text, str)):
            _result = GeminiService._translate_text(text, translation_instructions)

            if(hasattr(_result, "text")):
                return _result.text
            
            else:
                raise Exception("Unexpected error occurred. Please try again.")
        
        elif(isinstance(text, typing.Iterable)):
            
            _results = [GeminiService._translate_text(t, translation_instructions) for t in text]

            if(all(hasattr(_r, "text") for _r in _results)):
                return [_r.text for _r in _results]
            
        raise Exception("Unexpected state reached in gemini_translate.")

##-------------------start-of-gemini_translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    async def gemini_translate_async(text:typing.Union[str, typing.Iterable],
                                    override_previous_settings:bool = True,
                                    decorator:typing.Callable | None = None,
                                    translation_instructions:str | None = None,
                                    model:str="gemini-pro",
                                    temperature:float=0.5,
                                    top_p:float=0.9,
                                    top_k:int=40,
                                    stop_sequences:typing.List[str] | None=None,
                                    max_output_tokens:int | None=None) -> str | typing.List[str]:
        
        """

        Asynchronous version of gemini_translate().
        Will generally be faster for iterables. Order is preserved.

        Translates the given text using Gemini.

        This function assumes that the API key has already been set.

        Translation instructions default to translating the text to English. To change this, specify the instructions.

        This function is not for use for real-time translation, nor for generating multiple translation candidates. Another function may be implemented for this given demand.

        Parameters:
        text (string or iterable) : The text to translate.
        override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to this function.
        decorator (callable or None) : The decorator to use when translating. Typically for exponential backoff retrying.
        translation_instructions (string or None) : The translation instructions to use.
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

        _non_gemini_params = ["text", "override_previous_settings", "decorator", "translation_instructions"]

        for _key, _value in locals().items():
            if(_key not in _non_gemini_params):
                _settings[_key] = convert_to_correct_type(_key, _value)

        validate_easytl_translation_settings(_settings, "gemini")

        if(decorator != None):
            GeminiService._set_decorator(decorator)

        if(override_previous_settings == True):
            GeminiService._set_attributes(model=model,
                                          temperature=temperature,
                                          top_p=top_p,
                                          top_k=top_k,
                                          candidate_count=1,
                                          stream=False,
                                          stop_sequences=stop_sequences,
                                          max_output_tokens=max_output_tokens)
            
        if(isinstance(text, str)):
            _result = GeminiService._translate_text_async(text, translation_instructions)

            if(hasattr(_result, "text")):
                return _result.text # type: ignore
            
            else:
                raise Exception("Unexpected error occurred. Please try again.")
            
        elif(isinstance(text, typing.Iterable)):
            _tasks = [GeminiService._translate_text_async(_t, translation_instructions) for _t in text]
            _results = []

            for _future in asyncio.as_completed(_tasks):
                _result = await _future
                _results.append(_result)

            if(all(hasattr(_r, "text") for _r in _results)):
                return [_r.text for _r in _results]  # type: ignore
            
        raise Exception("Unexpected state reached in gemini_translate_async.")

##-------------------start-of-translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    def translate(text:str | typing.Iterable[str],
                  service:typing.Optional[typing.Literal["deepl", "openai", "gemini"]] = "deepl", 
                  **kwargs) -> typing.Union[typing.List[str], str]:
        
        """

        Translates the given text to the target language using the specified service.

        Please see the documentation for the specific translation function for the service you want to use.

        DeepL: deepl_translate()
        
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
            raise NotImplementedError("OpenAI service is not yet implemented.")

        else:
            raise NotImplementedError("Gemini service is not yet implemented.")
        
##-------------------start-of-translate_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    async def translate_async(text:str | typing.Iterable[str],
                              service:typing.Optional[typing.Literal["deepl", "openai", "gemini"]] = "deepl", 
                              **kwargs) -> typing.Union[typing.List[str], str]:
        
        """

        Asynchronous version of translate().
        
        Translates the given text to the target language using the specified service.

        Please see the documentation for the specific translation function for the service you want to use.

        DeepL: deepl_translate_async()

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
            raise NotImplementedError("OpenAI service is not yet implemented.")

        else:
            raise NotImplementedError("Gemini service is not yet implemented.")