## built-in libraries
import typing

## third-party libraries
from deepl.translator import Translator
from deepl.api_data import Language, SplitSentences, Formality, GlossaryInfo, TextResult

class DeepLService:

    api_key:str = ""
    
    translator:Translator

    target_lang:str | Language = "EN"
    source_lang:str | Language | None = None
    context:str | None = None
    split_sentences:typing.Literal["OFF", "ALL", "NO_NEWLINES"] |  SplitSentences | None = "ALL"
    preserve_formatting:bool | None = None
    formality:typing.Literal["default", "more", "less", "prefer_more", "prefer_less"] | Formality | None = None 
    glossary:str | GlossaryInfo | None = None
    tag_handling:typing.Literal["xml", "html"] | None = None
    outline_detection:bool | None = None
    non_splitting_tags:str | typing.List[str] | None = None
    splitting_tags:str | typing.List[str] | None = None
    ignore_tags:str | typing.List[str] | None = None

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def set_attributes(target_lang:str | Language = "EN",
                        source_lang:str | Language | None = None,
                        context:str | None = None,
                        split_sentences:typing.Literal["OFF", "ALL", "NO_NEWLINES"] |  SplitSentences | None = "ALL",
                        preserve_formatting:bool | None = None,
                        formality:typing.Literal["default", "more", "less", "prefer_more", "prefer_less"] | Formality | None = None,
                        glossary:str | GlossaryInfo | None = None,
                        tag_handling:typing.Literal["xml", "html"] | None = None,
                        outline_detection:bool | None = None,
                        non_splitting_tags:str | typing.List[str] | None = None,
                        splitting_tags:str | typing.List[str] | None = None,
                        ignore_tags:str | typing.List[str] | None = None) -> None:

        """

        Sets the attributes of the DeepL client.

        Parameters:
        target_lang (string or Language) : The target language to translate to.
        source_lang (string or Language or None) : The source language to translate from.
        context (string or None) : The context of the text.
        split_sentences (literal or SplitSentences or None) : The split sentences option.
        preserve_formatting (bool or None) : The preserve formatting option.
        formality (literal or Formality or None) : The formality option.
        glossary (string or GlossaryInfo or None) : The glossary option.
        tag_handling (literal or None) : The tag handling option.
        outline_detection (bool or None) : The outline detection option.
        non_splitting_tags (string or list - str or None) : The non-splitting tags option.
        splitting_tags (string or list - str or None) : The splitting tags option.
        ignore_tags (string or list - str or None) : The ignore tags option.

        """

        DeepLService.target_lang = target_lang
        DeepLService.source_lang = source_lang
        DeepLService.context = context
        DeepLService.split_sentences = split_sentences
        DeepLService.preserve_formatting = preserve_formatting
        DeepLService.formality = formality
        DeepLService.glossary = glossary
        DeepLService.tag_handling = tag_handling
        DeepLService.outline_detection = outline_detection
        DeepLService.non_splitting_tags = non_splitting_tags
        DeepLService.splitting_tags = splitting_tags
        DeepLService.ignore_tags = ignore_tags

##-------------------start-of-translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def translate(text:str) -> typing.Union[typing.List[TextResult], TextResult]:

        """

        Translates the given text to the target language.

        Parameters:
        text (string) : The text to translate.

        Returns:
        translation (TextResult) : The translation result.

        """

        isvalid, e = DeepLService.test_api_key_validity()

        if(not isvalid and e):
            raise e

        try:

            translation = DeepLService.translator.translate_text(text,
                                                                target_lang=DeepLService.target_lang,
                                                                source_lang=DeepLService.source_lang,
                                                                context=DeepLService.context,
                                                                split_sentences=DeepLService.split_sentences,
                                                                preserve_formatting=DeepLService.preserve_formatting,
                                                                formality=DeepLService.formality,
                                                                glossary=DeepLService.glossary,
                                                                tag_handling=DeepLService.tag_handling,
                                                                outline_detection=DeepLService.outline_detection,
                                                                non_splitting_tags=DeepLService.non_splitting_tags,
                                                                splitting_tags=DeepLService.splitting_tags,
                                                                ignore_tags=DeepLService.ignore_tags)

            return translation
        
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