## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import asyncio
import typing

## third-party libraries
from google.cloud import translate_v2 as translate
from google.cloud.translate_v2 import Client

from google.oauth2 import service_account
from google.oauth2.service_account import Credentials

## custom modules
from .util import _convert_iterable_to_str

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

    _log_directory:str | None = None


##-------------------start-of-_set_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_credentials(key_path:str):

        """
        
        Set the credentials for the Google Translate API.

        Parameters:
        key_path (str): The path to the JSON key file.

        """

        GoogleTLService._credentials = service_account.Credentials.from_service_account_file(key_path)


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
    

##-------------------start-of-_test_credential()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    @_redefine_client_decorator
    def _test_credential() -> typing.Tuple[bool, typing.Union[Exception, None]]:

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
    @_redefine_client_decorator
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


def main():
    credentials = service_account.Credentials.from_service_account_file(
        'C:\\Users\\Tetra\\Desktop\\LN\\API keys\\gen-lang-client-0189553349-6c305b2f0118.json')


    translate_client = translate.Client(credentials=credentials)

    text = 'Hello, world!'

    result = translate_client.translate(text, target_language='ru')

    print(result['translatedText'])

if __name__ == '__main__':
    main()