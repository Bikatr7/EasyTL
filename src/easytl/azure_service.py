## Copyright Alejandro (https://github.com/alemalvarez)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio
import uuid
import json
import time

# third-party libraries
import requests

## custom modules
from .util import _convert_iterable_to_str
from .decorators import _async_logging_decorator, _sync_logging_decorator
from .exceptions import EasyTLException

class AzureService:

    _api_key:str = ''
    _location:str = ''
    _endpoint:str = ''

    _path: str = '/translate'
    _api_version:str = '3.0'

    ## ISO 639-1 language codes
    _target_lang:str = 'en' # This can be a list of languages
    # Do we want to be possible to perform multiple translations at the same time?
    _source_lang:str | None = None

    _format:str = 'text'

    _semaphore_value:int = 15
    _semaphore:asyncio.Semaphore = asyncio.Semaphore(_semaphore_value)

    _rate_limit_delay:float | None = None

    _decorator_to_use:typing.Union[typing.Callable, None] = None

    _log_directory:str | None = None

##-------------------start-of-_set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_api_key(api_key: str): #
        """
        Sets the API key for the Azure service.

        Parameters:
        - api_key (str): The API key for accessing the Azure service.
        
        """
        # I changed this, so that the location and endpoint are part of the attributes instead of
        # of the credentials. Makes it easier to integrate in the core file.

        AzureService._api_key = api_key

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_attributes(target_language:str = 'en',
                        api_version:str = '3.0',
                        azure_region:str = "westus",
                        azure_endpoint:str = 'https://api.cognitive.microsofttranslator.com/',
                        source_language:str | None = None,
                        decorator:typing.Callable | None = None,
                        log_directory:str | None = None,
                        semaphore:int | None = None,
                        rate_limit_delay:float | None = None
                        ) -> None:
        """
        Sets the attributes of the Azure service.

        """
        ## API Attributes

        AzureService._target_lang = target_language
        AzureService._api_version = api_version
        AzureService._location = azure_region
        AzureService._endpoint = azure_endpoint
        AzureService._source_lang = source_language

        ## Service Attributes
        AzureService._decorator_to_use = decorator
        AzureService._log_directory = log_directory

        if(semaphore is not None):
            AzureService._semaphore_value = semaphore
            AzureService._semaphore = asyncio.Semaphore(semaphore)

        AzureService._rate_limit_delay = rate_limit_delay

##-------------------start-of-_prepare_translation_parameters()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------   

    @staticmethod
    def _prepare_translation_parameters():
        """
        Prepares the parameters for the translation request.
        """
        params = {
            'api-version': AzureService._api_version,
            'from': AzureService._source_lang,
            'to': [AzureService._target_lang]
        }

        return params
    
##-------------------start-of-_translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    @_sync_logging_decorator
    def _translate_text(text: str) -> typing.Union[typing.List[typing.Any], typing.Any]:

        """
        Translates the text using the Azure service.

        Parameters:
        - text (str): The text to translate.

        Returns:
        - response (list): The list of translations.
        """

    
        if (AzureService._decorator_to_use is None):
            return AzureService.__translate_text(text)
        
        
        decorated_function = AzureService._decorator_to_use(AzureService.__translate_text)
        return decorated_function(text)
        
    @staticmethod
    def __translate_text(text: str) -> typing.Union[typing.List[typing.Any], typing.Any]:

        """
        Translates the text using the Azure service.

        Parameters:
        - text (str): The text to translate.

        Returns:
        - response (list): The list of translations.
        """

        ## Prepare the translation parameters
        params = AzureService._prepare_translation_parameters()

        headers = {
            'Ocp-Apim-Subscription-Key': AzureService._api_key,
            'Ocp-Apim-Subscription-Region': AzureService._location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        if (isinstance(text, str)):
            body = [{   
                'text': text
            }]
        elif (isinstance(text, typing.Iterable)):
            body = []
            for t in text:
                body.append({
                    'text': t
                })

        try:
            url = AzureService._endpoint + AzureService._path

            request = requests.post(url, params=params, headers=headers, json=body)
            response = request.json()

            return response
        
        except Exception as _e:
            raise _e
        
##-------------------start-of-_translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    @_async_logging_decorator
    async def _translate_text_async(text: str) -> typing.Union[typing.List[typing.Any], typing.Any]:

        """
        Translates the text using the Azure service asynchronously.

        Parameters:
        - text (str): The text to translate.

        Returns:
        - response (list): The list of translations.
        """

        async with AzureService._semaphore:

            if (AzureService._rate_limit_delay is not None):
                await asyncio.sleep(AzureService._rate_limit_delay)

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: AzureService._translate_text(text))
            
##-------------------start-of-_test_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _test_credentials() -> typing.Tuple[bool, typing.Union[Exception, None]]:
        """
        Tests the credentials for the Azure service.

        Returns:
        - success (bool): True if the credentials are valid, False otherwise.
        - error (Exception): The error that occurred during the test.
        """

        try:
            AzureService._translate_text('Hola')
            return True, None

        except Exception as _e:
            return False, _e
        
##-------------------start-of-_calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _calculate_cost(text: str) -> typing.Tuple[int, float, str]:
            
            """
            Calculates the cost of the translation.
    
            Parameters:
            - text (str): The text to calculate the cost for.
    
            Returns:
            - cost (float): The cost of the translation.
            """

            # S1 Standard pricing:
            # $10 per million characters

            if(isinstance(text, typing.Iterable)):
                text = _convert_iterable_to_str(text)

            characters = len(text)
            cost = characters * 10 / 1000000
            model = "S1 Standard Pricing"
    
            return characters, cost, model