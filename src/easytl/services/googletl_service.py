## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import asyncio
import typing
import time

## third-party libraries
from google.cloud import translate_v2 as translate
from google.cloud.translate_v2 import Client

from google.oauth2 import service_account
from google.oauth2.service_account import Credentials

## custom modules
from ..util.util import _convert_iterable_to_str

class GoogleTLService:

    _translator:Client
    _credentials:Credentials

    ## ISO 639-1 language codes
    _target_lang:str = 'en'
    _source_lang:str | None = None
    
    _format:str = 'text'

    _semaphore_value:int = 15
    _semaphore:asyncio.Semaphore = asyncio.Semaphore(_semaphore_value)

    _rate_limit_delay:float | None = None

    _decorator_to_use:typing.Union[typing.Callable, None] = None

##-------------------start-of-_set_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_credentials(key_path:str):

        """
        
        Set the credentials for the Google Translate API.

        Parameters:
        key_path (str): The path to the JSON key file.

        """

        GoogleTLService._credentials = service_account.Credentials.from_service_account_file(key_path)

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _set_attributes(target_language:str = 'en',
                        format:str = 'text',
                        source_language:str | None = None,
                        decorator:typing.Callable | None = None,
                        semaphore:int | None = None,
                        rate_limit_delay:float | None = None
                        ) -> None:

        """

        Sets the attributes of the DeepL client.

        """

        ## API Attributes
    
        GoogleTLService._target_lang = target_language
        GoogleTLService._format = format
        GoogleTLService._source_lang = source_language

        ## Service Attributes

        GoogleTLService._decorator_to_use = decorator

        if(semaphore is not None):
            GoogleTLService._semaphore_value = semaphore
            GoogleTLService._semaphore = asyncio.Semaphore(semaphore)

        GoogleTLService._rate_limit_delay = rate_limit_delay

##-------------------start-of-_prepare_translation_parameters()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _prepare_translation_parameters(text:str):

        """

        Prepares the parameters for the translation.

        Parameters:
        text (string) : The text to translate.

        """

        params = {
            'values': text,
            'target_language': GoogleTLService._target_lang,
            'format_': GoogleTLService._format,
            'source_language': GoogleTLService._source_lang,
        }

        return params

##-------------------start-of-_redefine_client()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _redefine_client():

        """
        
        Redefine the Google Translate client with the new credentials.

        """

        GoogleTLService._translator = translate.Client(credentials=GoogleTLService._credentials)

##-------------------start-of-_redefine_client_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _redefine_client_decorator(func):

        """

        Wraps a function to redefine the GoogleTL client before doing anything that requires the client.

        Parameters:
        func (callable) : The function to wrap.

        Returns:
        wrapper (callable) : The wrapped function.

        """

        def wrapper(*args, **kwargs):
            GoogleTLService._redefine_client() 
            return func(*args, **kwargs)
        
        return wrapper

##-------------------start-of-translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _translate_text(text:str) -> typing.Union[typing.List[typing.Any], typing.Any]:

        """

        Translates the given text to the target language.

        Parameters:
        text (string) : The text to translate.

        Returns:
        translation (TextResult or list of TextResult) : The translation result.

        """

        ## decorators need to be applied outside of the function for reasons detailed in easytl.py

        if(GoogleTLService._rate_limit_delay is not None):
            time.sleep(GoogleTLService._rate_limit_delay)

        params = GoogleTLService._prepare_translation_parameters(text)

        try:

            return GoogleTLService._translator.translate(**params)
                        
        except Exception as _e:
            raise _e
        
##-------------------start-of-_translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
    @staticmethod
    async def _translate_text_async(text:str) -> typing.Union[typing.List[typing.Any], typing.Any]:

        """

        Translates the given text to the target language asynchronously.

        Parameters:
        text (string) : The text to translate.

        Returns:
        translation (TextResult or list of TextResult) : The translation result.

        """

        ## decorators need to be applied outside of the function for reasons detailed in easytl.py

        async with GoogleTLService._semaphore:

            if(GoogleTLService._rate_limit_delay is not None):
                await asyncio.sleep(GoogleTLService._rate_limit_delay)

            params = GoogleTLService._prepare_translation_parameters(text)

            try:
                
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, lambda: GoogleTLService._translator.translate(**params))
                
            except Exception as _e:
                raise _e
    
##-------------------start-of-_test_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    @_redefine_client_decorator
    def _test_credentials() -> typing.Tuple[bool, typing.Union[Exception, None]]:

        """

        Tests the validity of the credentials.

        Returns:
        validity (bool) : The validity of the credentials.
        e (Exception) : The exception that occurred, if any.

        """

        _validity = False

        try:

            _response = GoogleTLService._translator.translate('Hello, world!', target_language='ru')

            assert isinstance(_response['translatedText'], str) 

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

        ## $20.00 per 1,000,000 characters if already over 500,000 characters.
        ## We cannot check quota, due to api limitations.

        if(isinstance(text, typing.Iterable)):
            text = _convert_iterable_to_str(text)


        _number_of_characters = len(text)
        _cost = (_number_of_characters/1000000)*20.0
        _model = "google translate"

        return _number_of_characters, _cost, _model