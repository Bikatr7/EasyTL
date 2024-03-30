## built-in libraries
import typing

## third-party libraries
from deepl.translator import Translator
from .classes import Language, SplitSentences, Formality, GlossaryInfo, TextResult

class DeepLService:

    _api_key:str = ""
    
    _translator:Translator

    _target_lang:str | Language = "EN"
    _source_lang:str | Language | None = None
    _context:str | None = None
    _split_sentences:typing.Literal["OFF", "ALL", "NO_NEWLINES"] |  SplitSentences | None = "ALL"
    _preserve_formatting:bool | None = None
    _formality:typing.Literal["default", "more", "less", "prefer_more", "prefer_less"] | Formality | None = None 
    _glossary:str | GlossaryInfo | None = None
    _tag_handling:typing.Literal["xml", "html"] | None = None
    _outline_detection:bool | None = None
    _non_splitting_tags:str | typing.List[str] | None = None
    _splitting_tags:str | typing.List[str] | None = None
    _ignore_tags:str | typing.List[str] | None = None

    _decorator_to_use:typing.Union[typing.Callable, None] = None

##-------------------start-of-_set_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_decorator(decorator:typing.Callable) -> None:

        """

        Sets the decorator to use for the Gemini service. Should be a callable that returns a decorator.

        Parameters:
        decorator (callable) : The decorator to use.

        """

        DeepLService._decorator_to_use = decorator

##-------------------start-of-set_attributes()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _set_attributes(target_lang:str | Language = "EN",
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

        """

        DeepLService._target_lang = target_lang
        DeepLService._source_lang = source_lang
        DeepLService._context = context
        DeepLService._split_sentences = split_sentences
        DeepLService._preserve_formatting = preserve_formatting
        DeepLService._formality = formality
        DeepLService._glossary = glossary
        DeepLService._tag_handling = tag_handling
        DeepLService._outline_detection = outline_detection
        DeepLService._non_splitting_tags = non_splitting_tags
        DeepLService._splitting_tags = splitting_tags
        DeepLService._ignore_tags = ignore_tags

##-------------------start-of-translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _translate_text(text:str) -> typing.Union[typing.List[TextResult], TextResult]:

        """

        Translates the given text to the target language.

        Parameters:
        text (string) : The text to translate.

        Returns:
        translation (TextResult) : The translation result.

        """

        _is_valid, _e = DeepLService._test_api_key_validity()

        if(not _is_valid and _e):
            raise _e
        
        ## split sentences doesn't exactly match what deepl is expecting so..
        if(isinstance(DeepLService._split_sentences, str)):
            DeepLService._split_sentences = SplitSentences[DeepLService._split_sentences]

        try:

            if(DeepLService._decorator_to_use is None):
                return DeepLService._translator.translate_text(text,
                                                                target_lang=DeepLService._target_lang,
                                                                source_lang=DeepLService._source_lang,
                                                                context=DeepLService._context,
                                                                split_sentences=DeepLService._split_sentences,
                                                                preserve_formatting=DeepLService._preserve_formatting,
                                                                formality=DeepLService._formality,
                                                                glossary=DeepLService._glossary,
                                                                tag_handling=DeepLService._tag_handling,
                                                                outline_detection=DeepLService._outline_detection,
                                                                non_splitting_tags=DeepLService._non_splitting_tags,
                                                                splitting_tags=DeepLService._splitting_tags,
                                                                ignore_tags=DeepLService._ignore_tags)

            decorated_function = DeepLService._decorator_to_use(DeepLService._translate_text)

            return decorated_function(text,
                                        target_lang=DeepLService._target_lang,
                                        source_lang=DeepLService._source_lang,
                                        context=DeepLService._context,
                                        split_sentences=DeepLService._split_sentences,
                                        preserve_formatting=DeepLService._preserve_formatting,
                                        formality=DeepLService._formality,
                                        glossary=DeepLService._glossary,
                                        tag_handling=DeepLService._tag_handling,
                                        outline_detection=DeepLService._outline_detection,
                                        non_splitting_tags=DeepLService._non_splitting_tags,
                                        splitting_tags=DeepLService._splitting_tags,
                                        ignore_tags=DeepLService._ignore_tags)
    
        except Exception as _e:
            raise _e

##-------------------start-of-_set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _set_api_key(api_key:str) -> None:

        """

        Sets the API key for the DeepL client.

        Parameters:
        api_key (string) : The API key to set.

        """

        DeepLService._api_key = api_key

##-------------------start-of-_test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def _test_api_key_validity() -> typing.Tuple[bool, typing.Union[Exception, None]]:

        """

        Tests the validity of the API key.

        Returns:
        validity (bool) : True if the API key is valid, False if it is not.
        e (Exception) : The exception that was raised, if any.

        """

        _validity = False

        try:

            DeepLService._translator = Translator(DeepLService._api_key)

            DeepLService._translator.translate_text("私", target_lang="JA")

            _validity = True

            return _validity, None

        except Exception as _e:

            return _validity, _e
        
##-------------------start-of-_get_decorator()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _get_decorator() -> typing.Union[typing.Callable, None]:

        """

        Returns the decorator to use for the Gemini service.

        Returns:
        decorator (callable) : The decorator to use.

        """

        return DeepLService._decorator_to_use