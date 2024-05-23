## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

from .version import VERSION as __version__  # noqa

__author__ = "Bikatr7 <Tetralon07@gmail.com>"

from .easytl import EasyTL

from .classes import Language, SplitSentences, Formality, GlossaryInfo, TextResult
from .classes import Message, SystemTranslationMessage, ModelTranslationMessage
from .classes import ChatCompletion
from .classes import GenerateContentResponse, AsyncGenerateContentResponse, GenerationConfig
from .classes import AnthropicMessage, AnthropicTextBlock, AnthropicToolsBetaMessage, AnthropicToolUseBlock
from .classes import NOT_GIVEN, NotGiven

from .util import MODEL_COSTS, ALLOWED_GEMINI_MODELS, ALLOWED_OPENAI_MODELS, ALLOWED_ANTHROPIC_MODELS, VALID_JSON_OPENAI_MODELS, VALID_JSON_GEMINI_MODELS, VALID_JSON_ANTHROPIC_MODELS, MODEL_MAX_TOKENS

## deepL generic exception
from .exceptions import DeepLException

## google generic exception
from .exceptions import GoogleAPIError

## openai generic exception
from .exceptions import OpenAIError

## anthropic generic exception
from .exceptions import AnthropicError

## service specific exceptions
from .exceptions import OpenAIAPIError, OpenAIConflictError, OpenAINotFoundError, OpenAIAPIStatusError, OpenAIRateLimitError, OpenAIAPITimeoutError, OpenAIBadRequestError, OpenAIAPIConnectionError, OpenAIAuthenticationError, OpenAIInternalServerError, OpenAIPermissionDeniedError, OpenAIUnprocessableEntityError, OpenAIAPIResponseValidationError
from .exceptions import DeepLAuthorizationException, DeepLQuotaExceededException, DeepLConnectionException, DeepLTooManyRequestsException, DeepLDocumentNotReadyException, DeepLGlossaryNotFoundException, DeepLDocumentTranslationException
from .exceptions import GoogleAuthError, GoogleDefaultCredentialsError
from .exceptions import AnthropicAPIError, AnthropicConflictError, AnthropicNotFoundError, AnthropicAPIStatusError, AnthropicRateLimitError, AnthropicAPITimeoutError, AnthropicBadRequestError, AnthropicAPIConnectionError, AnthropicAuthenticationError, AnthropicInternalServerError, AnthropicPermissionDeniedError, AnthropicUnprocessableEntityError, AnthropicAPIResponseValidationError

__all__ = [
    "EasyTL",
    "Language", "SplitSentences", "Formality", "GlossaryInfo", "TextResult",
    "Message", "SystemTranslationMessage", "ModelTranslationMessage",
    "ChatCompletion",
    "GenerateContentResponse", "AsyncGenerateContentResponse", "GenerationConfig",
    "AnthropicMessage","AnthropicTextBlock", "AnthropicToolsBetaMessage", "AnthropicToolUseBlock",
    "NOT_GIVEN","NotGiven",
    "MODEL_COSTS", "ALLOWED_GEMINI_MODELS", "ALLOWED_OPENAI_MODELS", "ALLOWED_ANTHROPIC_MODELS", "VALID_JSON_OPENAI_MODELS", "VALID_JSON_GEMINI_MODELS", "VALID_JSON_ANTHROPIC_MODELS", "MODEL_MAX_TOKENS",
    "DeepLException",
    "GoogleAPIError",
    "OpenAIError",
    "AnthropicError",
    "OpenAIAPIError", "OpenAIConflictError", "OpenAINotFoundError", "OpenAIAPIStatusError", "OpenAIRateLimitError", "OpenAIAPITimeoutError", "OpenAIBadRequestError", "OpenAIAPIConnectionError", "OpenAIAuthenticationError", "OpenAIInternalServerError", "OpenAIPermissionDeniedError", "OpenAIUnprocessableEntityError", "OpenAIAPIResponseValidationError",
    "DeepLAuthorizationException", "DeepLQuotaExceededException", "DeepLConnectionException", "DeepLTooManyRequestsException", "DeepLDocumentNotReadyException", "DeepLGlossaryNotFoundException", "DeepLDocumentTranslationException",
    "GoogleAuthError", "GoogleDefaultCredentialsError",
    "AnthropicAPIError", "AnthropicConflictError", "AnthropicNotFoundError", "AnthropicAPIStatusError", "AnthropicRateLimitError", "AnthropicAPITimeoutError", "AnthropicBadRequestError", "AnthropicAPIConnectionError", "AnthropicAuthenticationError", "AnthropicInternalServerError", "AnthropicPermissionDeniedError", "AnthropicUnprocessableEntityError", "AnthropicAPIResponseValidationError"
]