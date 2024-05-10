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
from .util import _convert_iterable_to_st
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

    _logging_directory:str | None = None

##-------------------start-of-_set_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_credentials(api_key: str,
                         location: str = 'westus2',
                         endpoint: str = 'https://api.cognitive.microsofttranslator.com'):
        """
        Sets the credentials for the Azure service.

        Parameters:
        - api_key (str): The API key for accessing the Azure service.
        - location (str): The location of the Azure service. Default is 'westus2'.
        - endpoint (str): The endpoint URL of the Azure service. Default is 'https://api.cognitive.microsofttranslator.com'.
        """

        AzureService._api_key = api_key
        AzureService._location = location
        AzureService._endpoint = endpoint

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_attributes(target_language:str = 'en',
                        api_version:str = '3.0',
                        source_language:str | None = None,
                        decorator:typing.Callable | None = None,
                        logging_directory:str | None = None,
                        semaphore:int | None = None,
                        rate_limit_delay:float | None = None
                        ) -> None:
        """
        Sets the attributes of the Azure service.

        """
        ## API Attributes

        AzureService._target_lang = target_language
        AzureService._api_version = api_version
        AzureService._source_lang = source_language

        ## Service Attributes
        AzureService._decorator_to_use = decorator
        AzureService._logging_directory = logging_directory

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
    @_async_logging_decorator
    def _translate_text(text: str) -> typing.Union[typing.List[typing.Any], typing:Any]:

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

        body = [{
            'text': text
        }]

        try:
            if (AzureService._decorator_to_use is None):
                url = AzureService._endpoint + AzureService._path

                request = requests.post(url, params=params, headers=headers, json=body)
                response = request.json()

                return response
            
            else:
                decorated_function = AzureService._decorator_to_use(AzureService._translate_text)
                return decorated_function(text)
         
        except Exception as _e:
            raise _e