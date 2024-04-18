## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## deepl api data used by deepl_service to type check
from deepl.api_data import Language, SplitSentences, Formality, GlossaryInfo, TextResult

## openai api data used by openai_service to type check
from openai.types.chat.chat_completion import ChatCompletion

## gemini api data used by gemini_service to type check
from google.generativeai import GenerationConfig
from google.generativeai.types import GenerateContentResponse, AsyncGenerateContentResponse

##-------------------start-of-Message--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Message:

    """

    Message is a class that is used to send translation batches to the OpenAI API.

    """

    def __init__(self, content: str):
        self._content = content

    @property
    def role(self):
        raise NotImplementedError

    @property
    def content(self):
        return self._content
    
    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content
        }
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self):
        return f"<Message role={self.role} content='{self.content}'>"

##-------------------start-of-SystemTranslationMessage--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class SystemTranslationMessage(Message):

    """

    SystemTranslationMessage is a class that is used to send the system message to the OpenAI API.

    """

    @property
    def role(self):
        return 'system'

##-------------------start-of-ModelTranslationMessage--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ModelTranslationMessage(Message):

    """

    ModelTranslationMessage is a class that is used to send the model/user message to the OpenAI API.
    
    """
    
    @property
    def role(self):
        return 'user'