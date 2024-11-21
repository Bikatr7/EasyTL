## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio
import time


## third-party libraries
from deepl.translator import Translator

import deepl

## custom modules
from ..version import VERSION
from ..util.util import _convert_iterable_to_str
from ..classes import Language, SplitSentences, Formality, GlossaryInfo, TextResult

class DeepLService:

    _api_key:str = ""
    
    _translator:Translator

    _target_lang:str | Language = "EN-US"
    _source_lang:str | Language | None = None
    _context:str | None = None
    _split_sentences:typing.Literal["OFF", "ALL", "NO_NEWLINES"] |  SplitSentences | None = "ALL"
    _preserve_formatting:bool | None = None
    _formality:typing.Literal["default", "more", "less", "prefer_more", "prefer_less"] | Formality | None = None 
    _glossary:str | GlossaryInfo | None = None
    _tag_handling:typing.Literal["xml", "html"] | None = None
    _outline_detection:bool | None = None
    _non_splitting_tags:str | typing.List[str] | None = None
    _splitting_tags:str | typing.List[str] | None = None
    _ignore_tags:str | typing.List[str] | None = None

    _semaphore_value:int = 15
    _semaphore:asyncio.Semaphore = asyncio.Semaphore(_semaphore_value)

    _rate_limit_delay:float | None = None 

    _decorator_to_use:typing.Union[typing.Callable, None] = None

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _set_attributes(target_lang:str | Language = "EN",
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
                        ignore_tags:str | typing.List[str] | None = None,
                        decorator:typing.Callable | None = None,
                        semaphore:int | None = None,
                        rate_limit_delay:float | None = None
                        ) -> None:

        """

        Sets the attributes of the DeepL client.

        """

        ## API Attributes

        DeepLService._target_lang = target_lang
        DeepLService._source_lang = source_lang
        DeepLService._context = context
        DeepLService._split_sentences = split_sentences
        DeepLService._preserve_formatting = preserve_formatting
        DeepLService._formality = formality
        DeepLService._glossary = glossary
        DeepLService._tag_handling = tag_handling
        DeepLService._outline_detection = outline_detection
        DeepLService._non_splitting_tags = non_splitting_tags
        DeepLService._splitting_tags = splitting_tags
        DeepLService._ignore_tags = ignore_tags

        ## Service Attributes

        DeepLService._decorator_to_use = decorator

        ## if a decorator is used, we want to disable retries, otherwise set it to the default value which is 5
        if(DeepLService._decorator_to_use is not None):
            deepl.http_client.max_network_retries = 0

        else:
            deepl.http_client.max_network_retries = 5

        if(semaphore is not None):
            DeepLService._semaphore_value = semaphore
            DeepLService._semaphore = asyncio.Semaphore(semaphore)

        DeepLService._rate_limit_delay = rate_limit_delay

##-------------------start-of-_prepare_translation_parameters()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _prepare_translation_parameters(text:str):

        """

        Prepares the parameters for the translation.

        Parameters:
        text (string) : The text to translate.

        """

        if(isinstance(DeepLService._split_sentences, str)):
            DeepLService._split_sentences = SplitSentences[DeepLService._split_sentences]

        params = {
            "text": text,
            "target_lang": DeepLService._target_lang,
            "source_lang": DeepLService._source_lang,
            "context": DeepLService._context,
            "split_sentences": DeepLService._split_sentences,
            "preserve_formatting": DeepLService._preserve_formatting,
            "formality": DeepLService._formality,
            "glossary": DeepLService._glossary,
            "tag_handling": DeepLService._tag_handling,
            "outline_detection": DeepLService._outline_detection,
            "non_splitting_tags": DeepLService._non_splitting_tags,
            "splitting_tags": DeepLService._splitting_tags,
            "ignore_tags": DeepLService._ignore_tags,
        }

        return params

##-------------------start-of-translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _translate_text(text:str) -> typing.Union[typing.List[TextResult], TextResult]:

        """

        Translates the given text to the target language.

        Parameters:
        text (string) : The text to translate.

        Returns:
        translation (TextResult or list of TextResult) : The translation result.

        """


        ## decorators need to be applied outside of the function, for reasons detailed in easytl.py

        if(DeepLService._rate_limit_delay is not None):
            time.sleep(DeepLService._rate_limit_delay)

        params = DeepLService._prepare_translation_parameters(text)

        try:

            return DeepLService._translator.translate_text(**params)
            
        except Exception as _e:
            raise _e
        
##-------------------start-of-_translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    async def _translate_text_async(text:str) -> typing.Union[typing.List[TextResult], TextResult]:

        """

        Translates the given text to the target language asynchronously.

        Parameters:
        text (string) : The text to translate.

        Returns:
        translation (TextResult or list of TextResult) : The translation result.

        """

        ## decorators need to be applied outside of the function, for reasons detailed in easytl.py

        async with DeepLService._semaphore:

            if(DeepLService._rate_limit_delay is not None):
                await asyncio.sleep(DeepLService._rate_limit_delay)

            params = DeepLService._prepare_translation_parameters(text)

            try:

                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: DeepLService._translator.translate_text(**params))
                                
            except Exception as _e:
                raise _e

##-------------------start-of-_set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_api_key(api_key:str) -> None:

        """

        Sets the API key for the DeepL client.

        Parameters:
        api_key (string) : The API key to set.

        """

        DeepLService._api_key = api_key

##-------------------start-of-_test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
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

            DeepLService._translator = Translator(DeepLService._api_key).set_app_info("EasyTL", VERSION)

            DeepLService._translator.translate_text("私", target_lang="JA")

            _validity = True

            return _validity, None

        except Exception as _e:

            return _validity, _e
        
##-------------------start-of-_calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _calculate_cost(text:str | typing.Iterable) -> typing.Tuple[int, float, str]:

        """

        Calculates the cost of the translation.

        Parameters:
        text (string) : The text to calculate the cost for.

        Returns:
        cost (float) : The cost of the translation.

        """

        ## $25.00 per 1,000,000 characters if paid account, otherwise free under 500,000 characters per month.
        ## We cannot check quota, due to api limitations.

        if(isinstance(text, typing.Iterable)):
            text = _convert_iterable_to_str(text)


        _number_of_characters = len(text)
        _cost = (_number_of_characters/1000000)*25.0
        _model = "deepl"

        return _number_of_characters, _cost, _model