## built-in libraries
import typing

## custom modules
from easytl.deepl_service import DeepLService
from easytl.gemini_service import GeminiService
from easytl.openai_service import OpenAIService

from exceptions import DeepLException, 

##-------------------start-of-set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def set_api_key(api_type:typing.Literal["deepl", "gemini", "openai"], api_key:str) -> None:
    
    """

    Sets the API key for the specified API type.

    Parameters:
    api_type (string) : The API type to set the key for.
    api_key (string) : The API key to set.

    """

    if(api_type == "deepl"):
        DeepLService.set_api_key(api_key)

    elif(api_type == "gemini"):
        GeminiService.set_api_key(api_key)
        
    elif(api_type == "openai"):
        OpenAIService.set_api_key(api_key)

##-------------------start-of-test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
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
        is_valid, e = DeepLService.test_api_key_validity()

        if(is_valid == False):

            ## make sure issue is due to DeepL and not the fault of easytl, cause it needs to be raised if it is
            assert isinstance(e, DeepLException), e

            return False, e

    return True, None
    
    ## need to add the other services here
    ## but before that, need to convert said services to have non-asynchronous methods, as to not force the user to use async/await