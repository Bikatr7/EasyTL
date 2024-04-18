## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## deepL generic exception
from deepl.exceptions import DeepLException

## google generic exception
from google.api_core.exceptions import GoogleAPIError

## openai generic exception
from openai import OpenAIError

## service specific exceptions
from openai import AuthenticationError, InternalServerError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
from deepl.exceptions import AuthorizationException, QuotaExceededException
from google.auth.exceptions import GoogleAuthError

class EasyTLException(Exception):

    """

    EasyTLException is the base exception class for all exceptions in the EasyTL package.

    """

    pass

##-------------------start-of-InvalidAPIKeyException--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class InvalidAPIKeyException(EasyTLException):

    """

    InvalidAPIKeyException is an exception that is raised when the API key is invalid.

    """

    def __init__(self, model_name:str) -> None:

        """

        Parameters:
        model_name (string) : The name of the model that the API key is invalid for.

        """

        self.message = f"The API key is invalid for the model {model_name}."

##-------------------start-of-InvalidEasyTLSettingsException--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class InvalidEasyTLSettingsException(EasyTLException):

    """

    InvalidEasyTLSettingsException is an exception that is raised when the settings provided to the EasyTL class are invalid

    """

    def __init__(self, message:str) -> None:

        """

        Parameters:
        message (string) : The message to display when the exception is raised.

        """

        self.message = message

##-------------------start-of-InvalidAPITypeException--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class InvalidAPITypeException(EasyTLException):

    """

    InvalidAPITypeException is an exception that is raised when the API type provided to the EasyTL class is invalid

    """

    def __init__(self, message:str) -> None:

        """

        Parameters:
        message (string) : The message to display when the exception is raised.

        """

        self.message = message

##-------------------start-of-InvalidResponseFormatException--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class InvalidResponseFormatException(EasyTLException):

    """

    InvalidResponseFormatException is an exception that is raised when the response format is invalid

    """

    def __init__(self, message:str) -> None:

        """

        Parameters:
        message (string) : The message to display when the exception is raised.

        """

        self.message = message

##-------------------start-of-InvalidTextInputException--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class InvalidTextInputException(EasyTLException):

    """

    InvalidTextInputException is an exception that is raised when the text input is invalid

    """

    def __init__(self, message:str) -> None:

        """

        Parameters:
        message (string) : The message to display when the exception is raised.

        """

        self.message = message