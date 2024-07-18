## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
from typing_extensions import override

import typing

## third-party libraries

## deepl api data used by deepl_service to type check
from deepl.api_data import Language, SplitSentences, Formality, GlossaryInfo, TextResult

## openai api data used by openai_service to type check
from openai.types.chat.chat_completion import ChatCompletion

## gemini api data used by gemini_service to type check
from google.generativeai import GenerationConfig
from google.generativeai.types import GenerateContentResponse, AsyncGenerateContentResponse

## anthropic api data used by anthropic_service to type check
from anthropic.types import Message as AnthropicMessage, TextBlock as AnthropicTextBlock, ToolUseBlock as AnthropicToolUseBlock

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
    
    def __add__(self, other:str):
        if(isinstance(other, str)):
            new_content = self._content + other
            return self.__class__(new_content)
        
        return NotImplemented

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

##-------------------start-of-NotGiven--------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

class NotGiven:

    """
    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    Used until PEP 0661 is accepted
    
    For example:

    ```py
    def get(timeout: Union[int, NotGiven, None] = NotGiven()) -> Response:
        ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method definition.
    ```
    """

    def __bool__(self) -> typing.Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()