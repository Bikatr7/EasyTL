## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import asyncio
import uuid
import time

# third-party libraries
import requests

## custom modules
from ..util.util import _convert_iterable_to_str
from ..exceptions import BadAzureRegionException

class AzureService:

    _api_key:str = ""
    _location:str = ""
    _endpoint:str = ""

    _path: str = '/translate'
    _api_version:str = '3.0'

    ## ISO 639-1 language codes
    _target_lang:str = 'en' # This can be a list of languages
    _source_lang:str | None = None

    _format:str = 'text'

    _semaphore_value:int = 15
    _semaphore:asyncio.Semaphore = asyncio.Semaphore(_semaphore_value)

    _rate_limit_delay:float | None = None

    _decorator_to_use:typing.Union[typing.Callable, None] = None

##-------------------start-of-_set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_api_key(api_key:str): 

        """
        Sets the API key for the Azure service.

        Parameters:
        api_key (str) : The API key for accessing the Azure service.
        
        """

        AzureService._api_key = api_key

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_attributes(target_language:str = "en",
                        api_version:str = "3.0",
                        azure_region:str = "global",
                        azure_endpoint:str = 'https://api.cognitive.microsofttranslator.com/',
                        source_language:str | None = None,
                        decorator:typing.Callable | None = None,
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

        headers = {
            'Ocp-Apim-Subscription-Key': AzureService._api_key,
            'Ocp-Apim-Subscription-Region': AzureService._location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        return params, headers
    
##-------------------start-of-_translate_text()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _translate_text(text:str) -> typing.Union[typing.List[typing.Any], typing.Any]:

        """

        Translates the text using the Azure service.

        Parameters:
        text (str): The text to translate.

        Returns:
        response (list): The list of translations.

        """

        if(AzureService._rate_limit_delay is not None):
            time.sleep(AzureService._rate_limit_delay)

        params, headers = AzureService._prepare_translation_parameters()

        try:

            body = [{   
                'text': text
            }]

            url = AzureService._endpoint + AzureService._path

            request = requests.post(url, params=params, headers=headers, json=body)
            response = request.json()

            if('error' in response):
                raise BadAzureRegionException(f"{response['error']['message']}\n\nTip: Double check API key, region and endpoint :)")

            return response

        except Exception as _e:
            raise _e
        
##-------------------start-of-_translate_text_async()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def _translate_text_async(text:str) -> typing.Union[typing.List[typing.Any], typing.Any]:

        """

        Translates the text using the Azure service asynchronously.

        Parameters:
        text (str): The text to translate.

        Returns:
        response (list): The list of translations.

        """

        async with AzureService._semaphore:

            if(AzureService._rate_limit_delay is not None):
                await asyncio.sleep(AzureService._rate_limit_delay)

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: AzureService._translate_text(text))
            
##-------------------start-of-_test_credentials()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _test_credentials(azure_region: str) -> typing.Tuple[bool, typing.Union[Exception, None]]:

        """

        Tests the credentials for the Azure service.

        Returns:
        success (bool): True if the credentials are valid, False otherwise.
        error (Exception): The error that occurred during the test.

        """

        AzureService._set_attributes(azure_region=azure_region)

        try:
            AzureService._translate_text('Hola')
            return True, None

        except Exception as _e:
            return False, _e
        
##-------------------start-of-_calculate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _calculate_cost(text:str | typing.Iterable[str]) -> typing.Tuple[int, float, str]:
        
        """

        Calculates the cost of the translation.

        Parameters:
        text (str): The text to calculate the cost for.

        Returns:
        cost (float): The cost of the translation.

        """

        # S1 Standard pricing:
        # $10 per million characters

        if(isinstance(text, typing.Iterable)):
            text = _convert_iterable_to_str(text)

        characters = len(text)
        cost = characters * 10 / 1000000
        model = "S1 Standard Pricing"

        return characters, cost, model