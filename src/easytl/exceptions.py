## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## deepL generic exception
from deepl.exceptions import DeepLException

## google generic exception
from google.api_core.exceptions import GoogleAPIError

## openai generic exception
from openai import OpenAIError

## anthropic generic exception
from anthropic import AnthropicError

## azure generic exception
from requests import RequestException

## service specific exceptions
from openai import APIError as OpenAIAPIError, ConflictError as OpenAIConflictError, NotFoundError as OpenAINotFoundError, APIStatusError as OpenAIAPIStatusError, RateLimitError as OpenAIRateLimitError, APITimeoutError as OpenAIAPITimeoutError, BadRequestError as OpenAIBadRequestError, APIConnectionError as OpenAIAPIConnectionError, AuthenticationError as OpenAIAuthenticationError, InternalServerError as OpenAIInternalServerError, PermissionDeniedError as OpenAIPermissionDeniedError, UnprocessableEntityError as OpenAIUnprocessableEntityError, APIResponseValidationError as OpenAIAPIResponseValidationError
from deepl.exceptions import AuthorizationException as DeepLAuthorizationException, QuotaExceededException as DeepLQuotaExceededException, ConnectionException as DeepLConnectionException, TooManyRequestsException as DeepLTooManyRequestsException, DocumentNotReadyException as DeepLDocumentNotReadyException, GlossaryNotFoundException as DeepLGlossaryNotFoundException, DocumentTranslationException as DeepLDocumentTranslationException
from google.auth.exceptions import GoogleAuthError, DefaultCredentialsError as GoogleDefaultCredentialsError
from anthropic import APIError as AnthropicAPIError, ConflictError as AnthropicConflictError, NotFoundError as AnthropicNotFoundError, APIStatusError as AnthropicAPIStatusError, RateLimitError as AnthropicRateLimitError, APITimeoutError as AnthropicAPITimeoutError, BadRequestError as AnthropicBadRequestError, APIConnectionError as AnthropicAPIConnectionError, AuthenticationError as AnthropicAuthenticationError, InternalServerError as AnthropicInternalServerError, PermissionDeniedError as AnthropicPermissionDeniedError, UnprocessableEntityError as AnthropicUnprocessableEntityError, APIResponseValidationError as AnthropicAPIResponseValidationError

##-------------------start-of-EasyTLException--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

##-------------------start-of-TooManyInputTokensException--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class TooManyInputTokensException(EasyTLException):

    """

    TooManyInputTokensException is an exception that is raised when the input text contains too many tokens to be accepted by the API

    """

    def __init__(self, message:str) -> None:

        """

        Parameters:
        message (string) : The message to display when the exception is raised.

        """

        self.message = message

##-------------------start-of-BadAzureRegionException--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class BadAzureRegionException(EasyTLException):

    """

    BadAzureRegionException is an exception that is raised when the Azure region is invalid.

    """

    def __init__(self, message:str) -> None:

        """

        Parameters:
        message (string) : The message to display when the exception is raised.

        """

        self.message = message