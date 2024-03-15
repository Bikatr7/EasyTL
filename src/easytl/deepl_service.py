## built-in libraries
import typing

## third-party libraries
from deepl.translator import Translator
from deepl.api_data import Language, SplitSentences, Formality, GlossaryInfo

class DeepLService:

    api_key:str
    translator:Translator

    target_lang:str | Language | None
    source_lang:str | Language | None
    context:str | None
    split_sentences:typing.Literal["OFF", "ALL", "NO_NEWLINES"] |  SplitSentences | None = "ALL"
    preserve_formatting:bool | None
    formality:typing.Literal["default", "more", "less", "prefer_more", "prefer_less"] | Formality | None = None
    glossary:


##-------------------start-of-translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def translate(text:str, target_lang:str, source_lang:str) -> str:

        """

        Translates the text to the target language.

        Parameters:
        text (string) : The text to translate.
        target_lang (string) : The target language.
        source_lang (string) : The source language.

        Returns:
        translation (string) : The translated text.

        """

        isvalid, e = DeepLService.test_api_key_validity()

        if(not isvalid and e):
            raise e

        try:

            translation = DeepLService.translator.translate_text(text, target_lang=target_lang, source_lang=source_lang)

            return str(translation)
        
        except Exception as e:
            raise e

##-------------------start-of-set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def set_api_key(api_key:str) -> None:

        """

        Sets the API key for the DeepL client.

        Parameters:
        api_key (string) : The API key to set.

        """

        DeepLService.api_key = api_key

##-------------------start-of-test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def test_api_key_validity() -> typing.Tuple[bool, typing.Union[Exception, None]]:

        """

        Tests the validity of the API key.

        Returns:
        validity (bool) : True if the API key is valid, False if it is not.
        e (Exception) : The exception that was raised, if any.

        """

        validity = False

        try:

            DeepLService.translator = Translator(DeepLService.api_key)

            DeepLService.translator.translate_text("test", target_lang="JA")

            validity = True

            return validity, None

        except Exception as e:

            return validity, e