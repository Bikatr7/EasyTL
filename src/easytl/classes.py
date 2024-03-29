## deepl api data used by deepl_service to type check
from deepl.api_data import Language, SplitSentences, Formality, GlossaryInfo, TextResult

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