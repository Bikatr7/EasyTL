## built-in libraries
import typing

## third-party libraries
from deepl.api_data import Language, SplitSentences, Formality, GlossaryInfo, TextResult

## custom modules
from easytl.deepl_service import DeepLService
from easytl.gemini_service import GeminiService
from easytl.openai_service import OpenAIService

from .exceptions import DeepLException

##-------------------start-of-set_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def set_api_key(api_type:typing.Literal["deepl", "gemini", "openai"], api_key:str) -> None:
    
    """

    Sets the API key for the specified API type.

    Parameters:
    api_type (string) : The API type to set the key for.
    api_key (string) : The API key to set.

    """

    if(api_type == "deepl"):
        DeepLService.set_api_key(api_key)

    elif(api_type == "gemini"):
        GeminiService.set_api_key(api_key)
        
    elif(api_type == "openai"):
        OpenAIService.set_api_key(api_key)

##-------------------start-of-test_api_key_validity()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
def test_api_key_validity(api_type:typing.Literal["deepl", "gemini", "openai"]) -> typing.Tuple[bool, typing.Optional[Exception]]:
    
    """

    Tests the validity of the API key for the specified API type.

    Parameters:
    api_type (string) : The API type to test the key for.

    Returns:
    (bool) : Whether the API key is valid.
    (Exception) : The exception that was raised, if any. None otherwise.

    """

    if(api_type == "deepl"):
        is_valid, e = DeepLService.test_api_key_validity()

        if(is_valid == False):

            ## make sure issue is due to DeepL and not the fault of easytl, cause it needs to be raised if it is
            assert isinstance(e, DeepLException), e

            return False, e

    return True, None
    
    ## need to add the other services here
    ## but before that, need to convert said services to have non-asynchronous methods, as to not force the user to use async/await

##-------------------start-of-deepl_translate()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def deepl_translate(text:typing.Union[str, typing.Iterable],
                    target_lang:str | Language = "EN",
                    override_previous_settings:bool = True,
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
                    ignore_tags:str | typing.List[str] | None = None) -> typing.Union[typing.List[str], str]:
    
    """

    Translates the given text to the target language using DeepL.

    This function assumes that the API key has already been set.

    Parameters:
    text (string or iterable) : The text to translate.
    target_lang (string or Language) : The target language to translate to.
    override_previous_settings (bool) : Whether to override the previous settings that were used during the last call to this function.
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

    Returns:
    translation (TextResult or list - TextResult) : The translation result.

    """

    if(override_previous_settings == True):
        DeepLService.set_attributes(target_lang, 
                                    source_lang, 
                                    context, 
                                    split_sentences,
                                    preserve_formatting, 
                                    formality, 
                                    glossary, tag_handling, outline_detection, non_splitting_tags, splitting_tags, ignore_tags)
        
    if(isinstance(text, str)):
        return DeepLService.translate(text).text # type: ignore
    
    else:
        return [DeepLService.translate(t).text for t in text] # type: ignore