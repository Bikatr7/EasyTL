## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import json
import logging
import typing

## third-party libraries
import tiktoken

## custom modules
from ..util.constants import ALLOWED_OPENAI_MODELS, ALLOWED_GEMINI_MODELS, ALLOWED_ANTHROPIC_MODELS, MODEL_MAX_TOKENS
from ..util.util import _convert_iterable_to_str, _convert_to_correct_type, _update_model_name

from ..exceptions import InvalidEasyTLSettingsException, TooManyInputTokensException
from ..classes import ModelTranslationMessage, NotGiven, NOT_GIVEN

##-------------------start-of-_return_curated_anthropic_settings()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _return_curated_anthropic_settings(local_settings:dict[str, typing.Any]) -> dict:

    """

    Returns the curated Anthropic settings.

    What this does is it takes local_settings from the calling function, and then returns a dictionary with the settings that are relevant to Anthropic that were converted to the correct type.

    """

    _settings = {
    "anthropic_model": "",
    "anthropic_temperature": "",
    "anthropic_top_p": "",
    "anthropic_top_k": "",
    "anthropic_stop_sequences": "",
    "anthropic_max_output_tokens": "",
    }

    _non_anthropic_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "response_type", "semaphore", "translation_delay"]
    _custom_validation_params = ["anthropic_stop_sequences", "anthropic_response_schema"]

    for _key in _settings.keys():
        param_name = _key.replace("anthropic_", "")
        if(param_name in local_settings and _key not in _non_anthropic_params and _key not in _custom_validation_params):
            _settings[_key] = _convert_to_correct_type(_key, local_settings[param_name])

    return _settings

##-------------------start-of-_return_curated_gemini_settings()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _return_curated_gemini_settings(local_settings:dict[str, typing.Any]) -> dict:

    """
    
    Returns the curated Gemini settings.

    What this does is it takes local_settings from the calling function, and then returns a dictionary with the settings that are relevant to Gemini that were converted to the correct type.
    
    """

    _settings = {
    "gemini_model": "",
    "gemini_temperature": "",
    "gemini_top_p": "",
    "gemini_top_k": "",
    "gemini_stop_sequences": "",
    "gemini_max_output_tokens": ""
    }

    _non_gemini_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "response_type", "semaphore", "translation_delay"]
    _custom_validation_params = ["gemini_stop_sequences", "gemini_response_schema"]

    for _key in _settings.keys():
        param_name = _key.replace("gemini_", "")
        if(param_name in local_settings and _key not in _non_gemini_params and _key not in _custom_validation_params):
            _settings[_key] = _convert_to_correct_type(_key, local_settings[param_name])

    return _settings

##-------------------start-of-_return_curated_openai_settings()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _return_curated_openai_settings(local_settings:dict[str, typing.Any]) -> dict:
        
        """
        
        Returns the curated OpenAI settings.

        What this does is it takes local_settings from the calling function, and then returns a dictionary with the settings that are relevant to OpenAI that were converted to the correct type.

        """

        _settings = {
        "openai_model": "",
        "openai_temperature": "",
        "openai_top_p": "",
        "openai_stop": "",
        "openai_max_tokens": "",
        "openai_presence_penalty": "",
        "openai_frequency_penalty": ""
        }

        _non_openai_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "response_type", "response_schema", "semaphore", "translation_delay"]
        _custom_validation_params = ["openai_stop", "openai_response_schema"]

        for _key in _settings.keys():
            param_name = _key.replace("openai_", "")
            if(param_name in local_settings and _key not in _non_openai_params and _key not in _custom_validation_params):
                _settings[_key] = _convert_to_correct_type(_key, local_settings[param_name])

        return _settings

##-------------------start-of-_return_curated_gemini_settings()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _validate_stop_sequences(stop_sequences:typing.List[str] | None | NotGiven) -> None:

    assert stop_sequences in [None, NOT_GIVEN] or isinstance(stop_sequences, str) or (hasattr(stop_sequences, '__iter__') and all(isinstance(i, str) for i in stop_sequences)), InvalidEasyTLSettingsException("Invalid stop sequences. Must be a string or a list of strings.") # type: ignore

##-------------------start-of-_validate_response_schema()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _validate_response_schema(response_schema:str | typing.Mapping[str, typing.Any] | None = None) -> typing.Mapping[str, typing.Any] | None:

    if(response_schema is None):
        return None

    if(isinstance(response_schema, str)):
        try:
            return json.loads(response_schema)
        except json.JSONDecodeError:
            raise InvalidEasyTLSettingsException("Invalid response_schema. Must be a valid JSON string or None.")

    if(isinstance(response_schema, dict)):

        try:
            json.dumps(response_schema)
            return response_schema
        
        except (TypeError, OverflowError):
            raise InvalidEasyTLSettingsException("Invalid response_schema. Must be a valid JSON object or None.")

    raise InvalidEasyTLSettingsException("Invalid response_schema. Must be a valid JSON, a valid JSON string, or None.")


##-------------------start-of-validate_text_length()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _validate_text_length(text:str | typing.Iterable[str] | ModelTranslationMessage | typing.Iterable[ModelTranslationMessage] , model:str, service:str) -> None:

    """

    Validates the length of the input text.

    Parameters:
    text (string | typing.Iterable[string]) : The text to validate the length of.
    model (string) : The model to validate the text length for.
    service (string) : The service to validate the text length for.

    """

    try:

        if(isinstance(text, ModelTranslationMessage)):
            text = str(text)

        text = _convert_iterable_to_str(text) 

        if(service == "openai"):

            ## this just gets the latest model if they passed in a generic model name
            model = _update_model_name(model)
            
            _encoding = tiktoken.encoding_for_model(model)
            _num_tokens = len(_encoding.encode(text))

            _max_tokens_allowed = MODEL_MAX_TOKENS.get(model, {}).get("max_input_tokens")

            ## silently return if the model is not in the list of models with a max token limit
            if(not _max_tokens_allowed):
                return

            ## we can do a hard error with openai since we can accurately count tokens
            if(_num_tokens > _max_tokens_allowed):
                raise TooManyInputTokensException(f"Input text exceeds the maximum token limit of {model}.")
            
        else:
            _encoding = tiktoken.encoding_for_model("gpt-4-turbo-2024-04-09")
            _num_tokens = len(_encoding.encode(text))

            _max_tokens_allowed = MODEL_MAX_TOKENS.get(model, {}).get("max_input_tokens")

            ## silently return if the model is not in the list of models with a max token limit
            if(not _max_tokens_allowed):
                return

            ## we can't accurately count tokens with gemini/anthropic, so we'll just do a warning
            if(_num_tokens > _max_tokens_allowed):
                logging.warning(f"Input text may exceed the maximum token limit of {model}.")

    except TooManyInputTokensException:
        raise

    ## soft error, pretty sure this thing will break randomly
    except Exception as e:
        logging.error(f"Error validating text length: {str(e)}")

##-------------------start-of-_validate_easytl_llm_translation_settings()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _validate_easytl_llm_translation_settings(settings:dict, type:typing.Literal["gemini","openai", "anthropic"]) -> None:

    """

    Validates the Kijiku Rules.json file.

    Parameters:
    settings (dict) : The settings to validate.
    type (typing.Literal["gemini","openai"]) : The type of settings to validate.

    """

    ## Commented out keys are not used in the current version of EasyTL, but are kept.
    ## Stuff like stop, logit_bias, and n/candidate_count are not in use because there is simply no need for them in EasyTL.
    ## Stream may be used in the future, but is not used in the current version of EasyTL.
    ## They are typically hardcoded by EasyTL.

    ## The exception is openai_stop, and gemini_stop_sequences, which aren't validated here, rather in easytl.py, but still used and given to the model.

    _openai_keys = [
        "openai_model",
        "openai_temperature",
        "openai_top_p",
      #  "openai_n",
    #    "openai_stream",
     #   "openai_stop",
      ##  "openai_logit_bias",
        "openai_max_tokens",
        "openai_presence_penalty",
        "openai_frequency_penalty"
    ]

    _gemini_keys = [
        "gemini_model",
        "gemini_temperature",
        "gemini_top_p",
        "gemini_top_k",
    ##    "gemini_candidate_count",
    ##    "gemini_stream",
  ##      "gemini_stop_sequences",
        "gemini_max_output_tokens"
    ]

    _anthropic_keys = [
        "anthropic_model",
        "anthropic_temperature",
        "anthropic_top_p",
        "anthropic_top_k",
    ##    "anthropic_stream",
    ##    "anthropic_stop_sequences",
        "anthropic_max_output_tokens"
    ]

    _validation_rules = {
        "openai_model": lambda x: isinstance(x, str) and x in ALLOWED_OPENAI_MODELS or x is None or x is NOT_GIVEN,
        "openai_temperature": lambda x: isinstance(x, (int, float)) and 0 <= x <= 2 or x is None or x is NOT_GIVEN,
        "openai_top_p": lambda x: isinstance(x, (int, float)) and 0 <= x <= 1 or x is None or x is NOT_GIVEN,
        "openai_max_tokens": lambda x: x is None or x is NOT_GIVEN or (isinstance(x, int) and x > 0),
        "openai_presence_penalty": lambda x: isinstance(x, (int, float)) and -2 <= x <= 2 or x is None or x is NOT_GIVEN,
        "openai_frequency_penalty": lambda x: isinstance(x, (int, float)) and -2 <= x <= 2 or x is None or x is NOT_GIVEN,
        "gemini_model": lambda x: isinstance(x, str) and x in ALLOWED_GEMINI_MODELS or x is None or x is NOT_GIVEN,
        "gemini_prompt": lambda x: x not in ["", "None", None, NOT_GIVEN],
        "gemini_temperature": lambda x: isinstance(x, (int, float)) and 0 <= x <= 2 or x is None or x is NOT_GIVEN,
        "gemini_top_p": lambda x: x is None or x is NOT_GIVEN or (isinstance(x, (int, float)) and 0 <= x <= 2),
        "gemini_top_k": lambda x: x is None or x is NOT_GIVEN or (isinstance(x, int) and x >= 0),
        "gemini_max_output_tokens": lambda x: x is None or x is NOT_GIVEN or isinstance(x, int),
        "anthropic_model": lambda x: isinstance(x, str) and x in ALLOWED_ANTHROPIC_MODELS or x is None or x is NOT_GIVEN,
        "anthropic_temperature": lambda x: isinstance(x, (int, float)) and 0 <= x <= 1 or x is None or x is NOT_GIVEN,
        "anthropic_top_p": lambda x: isinstance(x, (int, float)) and 0 <= x <= 1 or x is None or x is NOT_GIVEN,
        "anthropic_top_k": lambda x: isinstance(x, int) and x > 0 or x is None or x is NOT_GIVEN,
        "anthropic_max_output_tokens": lambda x: x is None or x is NOT_GIVEN or (isinstance(x, int) and x > 0)
    }
    
    try:

        ## assign to variables to reduce repetitive access    
        if(type == "openai"):


            ## ensure all keys are present
            assert all(_key in settings for _key in _openai_keys)

            ## validate each _key using the validation rules
            for _key, _validate in _validation_rules.items():
                if(_key in settings and not _validate(settings[_key])):
                    raise ValueError(f"Invalid _value for {_key}")
                
      ##      settings["openai_logit_bias"] = None
       ##     settings["openai_stream"] = False
   ##         settings["openai_n"] = 1

        elif(type == "gemini"):

            ## ensure all keys are present
            assert all(_key in settings for _key in _gemini_keys)

            ## _validate each _key using the validation rules
            for _key, _validate in _validation_rules.items():
                if (_key in settings and not _validate(settings[_key])):
                    raise ValueError(f"Invalid _value for {_key}")
                
        ##    settings["gemini_stream"] = False
      ##      settings["gemini_candidate_count"] = 1

        elif(type == "anthropic"):

            ## ensure all keys are present
            assert all(_key in settings for _key in _anthropic_keys)

            ## _validate each _key using the validation rules
            for _key, _validate in _validation_rules.items():
                if (_key in settings and not _validate(settings[_key])):
                    raise ValueError(f"Invalid _value for {_key}")
        
    except Exception as e:
        raise InvalidEasyTLSettingsException(f"Invalid settings, Due to: {str(e)}")