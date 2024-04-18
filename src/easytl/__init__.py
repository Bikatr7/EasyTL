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

from .util import MODEL_COSTS, ALLOWED_GEMINI_MODELS, ALLOWED_OPENAI_MODELS, VALID_JSON_OPENAI_MODELS

from .exceptions import DeepLException, GoogleAPIError, OpenAIError, EasyTLException, InvalidAPIKeyException, InvalidEasyTLSettingsException, InvalidTextInputException, InvalidAPITypeException, InvalidResponseFormatException
from .exceptions import AuthenticationError, InternalServerError, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
from .exceptions import AuthorizationException, QuotaExceededException
from .exceptions import GoogleAuthError