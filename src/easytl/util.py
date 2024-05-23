## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing
import json
import logging

## third-party libraries
import tiktoken
import backoff

## custom modules
import google.generativeai as genai

from .exceptions import InvalidEasyTLSettingsException, GoogleAPIError, TooManyInputTokensException
from .classes import NotGiven, NOT_GIVEN, ModelTranslationMessage

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

    _non_anthropic_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "logging_directory", "response_type", "semaphore", "translation_delay"]
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

    _non_gemini_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "logging_directory", "response_type", "semaphore", "translation_delay"]
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

        _non_openai_params = ["text", "override_previous_settings", "decorator", "translation_instructions", "logging_directory", "response_type", "response_schema", "semaphore", "translation_delay"]
        _custom_validation_params = ["openai_stop"]

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
            _num_tokens = _gemini_count_tokens(text, model)

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

##-------------------start-of-_string_to_bool()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _string_to_bool(string:str) -> bool:

    return string.lower() in ['true', '1', 'yes', 'y', 't']

##-------------------start-of-_convert_iterable_to_str()-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _convert_iterable_to_str(iterable:typing.Iterable) -> str:

    if(isinstance(iterable, str)):
        return iterable

    return "".join(map(str, iterable))

##-------------------start-of-is_iterable_of_strings()-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _is_iterable_of_strings(value):

    if(isinstance(value, str)):
        return False
    
    try:
        _iterator = iter(value)
        
    except TypeError:
        return False
    
    return all(isinstance(_item, str) for _item in _iterator)

##-------------------start-of-_gemini_count_tokens()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@backoff.on_exception(backoff.expo, exception=(GoogleAPIError), logger=logging.getLogger(), max_tries=50, raise_on_giveup=True)
def _gemini_count_tokens(text:str, model:str="gemini-pro") -> int:

    """

    Counts the number of tokens in the given text.

    Parameters:
    text (string) : The text to count tokens in.

    Returns:
    total_tokens (int) : The number of tokens in the text.

    """

    return genai.GenerativeModel("gemini-pro").count_tokens(text).total_tokens

##-------------------start-of-_validate_easytl_translation_settings()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _validate_easytl_translation_settings(settings:dict, type:typing.Literal["gemini","openai", "anthropic"]) -> None:

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

##-------------------start-of-_convert_to_correct_type()-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _convert_to_correct_type(setting_name:str, initial_value:str) -> typing.Any:

    """

    Converts the input string to the correct _type based on the setting name.

    Parameters:
    setting_name (str) : The name of the setting to convert.
    initial_value (str) : The initial value to convert.

    Returns:
    (typing.Any) : The converted value.

    """

    _value = initial_value
    
    _type_expectations = {
        "openai_model": {"_type": str, "constraints": lambda x: x in ALLOWED_OPENAI_MODELS or x is None or x is NOT_GIVEN},
        ## "openai_system_message": {"_type": str, "constraints": lambda x: x not in ["", "None", None, NOT_GIVEN]},
        "openai_temperature": {"_type": float, "constraints": lambda x: isinstance(x, (int, float)) and 0 <= x <= 2 or x is None or x is NOT_GIVEN},
        "openai_top_p": {"_type": float, "constraints": lambda x: isinstance(x, (int, float)) and 0 <= x <= 2 or x is None or x is NOT_GIVEN},
        ## "openai_n": {"_type": int, "constraints": lambda x: x == 1},
        ## "openai_stream": {"_type": bool, "constraints": lambda x: x is False},
        ## "openai_stop": {"_type": None, "constraints": lambda x: x is None or x is NOT_GIVEN},
        ## "openai_logit_bias": {"_type": None, "constraints": lambda x: x is None or x is NOT_GIVEN},
        "openai_max_tokens": {"_type": int, "constraints": lambda x: x is None or x is NOT_GIVEN or (isinstance(x, int) and x > 0)},
        "openai_presence_penalty": {"_type": float, "constraints": lambda x: isinstance(x, (int, float)) and -2 <= x <= 2 or x is None or x is NOT_GIVEN},
        "openai_frequency_penalty": {"_type": float, "constraints": lambda x: isinstance(x, (int, float)) and -2 <= x <= 2 or x is None or x is NOT_GIVEN},
        "gemini_model": {"_type": str, "constraints": lambda x: x in ALLOWED_GEMINI_MODELS or x is None or x is NOT_GIVEN},
        ## "gemini_prompt": {"_type": str, "constraints": lambda x: x not in ["", "None", None, NOT_GIVEN]},
        "gemini_temperature": {"_type": float, "constraints": lambda x: isinstance(x, (int, float)) and 0 <= x <= 2 or x is None or x is NOT_GIVEN},
        "gemini_top_p": {"_type": float, "constraints": lambda x: x is None or x is NOT_GIVEN or (isinstance(x, float) and 0 <= x <= 2)},
        "gemini_top_k": {"_type": int, "constraints": lambda x: x is None or x is NOT_GIVEN or (isinstance(x, int) and x >= 0)},
        ## "gemini_candidate_count": {"_type": int, "constraints": lambda x: x == 1},
        ## "gemini_stream": {"_type": bool, "constraints": lambda x: x is False},
        ## "gemini_stop_sequences": {"_type": list, "constraints": lambda x: x is None or x is NOT_GIVEN or all(isinstance(i, str) for i in x)},
        "gemini_max_output_tokens": {"_type": int, "constraints": lambda x: x is None or x is NOT_GIVEN or isinstance(x, int)},
        "anthropic_model": {"_type": str, "constraints": lambda x: x in ALLOWED_ANTHROPIC_MODELS or x is None or x is NOT_GIVEN},
        "anthropic_temperature": {"_type": float, "constraints": lambda x: isinstance(x, (int, float)) and 0 <= x <= 1 or x is None or x is NOT_GIVEN},
        "anthropic_top_p": {"_type": float, "constraints": lambda x: isinstance(x, (int, float)) and 0 <= x <= 1 or x is None or x is NOT_GIVEN},
        "anthropic_top_k": {"_type": int, "constraints": lambda x: x is None or x is NOT_GIVEN or (isinstance(x, int) and x > 0)},
        ## "anthropic_stream": {"_type": bool, "constraints": lambda x: x is False},
        ## "anthropic_stop_sequences": {"_type": list, "constraints": lambda x: x is None or x is NOT_GIVEN or all(isinstance(i, str) for i in x)},
        "anthropic_max_output_tokens": {"_type": int, "constraints": lambda x: x is None or x is NOT_GIVEN or (isinstance(x, int) and x > 0)}
    }

    if(setting_name not in _type_expectations):
        raise ValueError("Invalid setting name")

    _setting_info = _type_expectations[setting_name]

    if("stream" in setting_name):
        _value = _string_to_bool(initial_value)

    elif(isinstance(initial_value, str) and initial_value.lower() in ["none","null"]):
        _value = None

    if(_setting_info["_type"] is None):
        _converted_value = None

    elif(_setting_info["_type"] == NOT_GIVEN or NotGiven):
        _converted_value = NOT_GIVEN

    elif(_setting_info["_type"] == int) or (_setting_info["_type"] == float):

        if(_value is None or _value == ''):
            _converted_value = None

        elif(_value == "NOT_GIVEN" or _value == NotGiven or _value == NOT_GIVEN):
            _converted_value = NOT_GIVEN
            
        elif(_setting_info["_type"] == int):
            _converted_value = int(_value)

        else:
            _converted_value = float(_value)

    else:
        _converted_value = _setting_info["_type"](_value)

    if("constraints" in _setting_info and not _setting_info["constraints"](_converted_value)):
        raise ValueError(f"{setting_name} out of range")

    return _converted_value

##-------------------start-of-_estimate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@staticmethod
def _estimate_cost(text:str | typing.Iterable, model:str, price_case:int | None = None) -> typing.Tuple[int, float, str]:

    """

    Attempts to estimate cost.

    Parameters:
    text (str | typing.Iterable) : the text to translate.
    model (string) : the model used to translate the text.
    price_case (int) : the price case used to calculate the cost.

    Returns:
    _num_tokens (int) : the number of tokens used.
    _min_cost (float) : the minimum cost of translation.
    model (string) : the model used to translate the text.

    """

    assert model in ALLOWED_OPENAI_MODELS + ALLOWED_GEMINI_MODELS + ALLOWED_ANTHROPIC_MODELS, f"""EasyTL does not support : {model}"""

    ## default models are first, then the rest are sorted by price case
    if(price_case is None):

        if(model == "gpt-3.5-turbo"):
            print("Warning: gpt-3.5-turbo may change over time. Estimating cost assuming gpt-3.5-turbo-0125 as it is the most recent version of gpt-3.5-turbo.")
            return _estimate_cost(text, model="gpt-3.5-turbo-0125")
        
        elif(model == "gpt-3.5-turbo-16k"):
            print("Warning: gpt-3.5-turbo-16k may change over time. Estimating cost assuming gpt-3.5-turbo-16k-0613 as it is the most recent version of gpt-3.5-turbo-16k.")
            return _estimate_cost(text, model="gpt-3.5-turbo-16k-0613")
        
        elif(model == "gpt-4"):
            print("Warning: gpt-4 may change over time. Estimating cost assuming gpt-4-0613 as it is the most recent version of gpt-4.")
            return _estimate_cost(text, model="gpt-4-0613")
        
        elif(model == "gpt-4-32k"):
            print("Warning: gpt-4-32k may change over time. Estimating cost assuming gpt-4-32k-0613 as it is the most recent version of gpt-4-32k.")
            return _estimate_cost(text, model="gpt-4-32k-0613")
        
        elif(model == "gpt-4-turbo"):
            print("Warning: gpt-4-turbo may change over time. Estimating cost assuming gpt-4-turbo-2024-04-09 as it is the most recent version of gpt-4-turbo.")
            return _estimate_cost(text, model="gpt-4-turbo-2024-04-09")
                
        elif(model == "gpt-4-turbo-preview"):
            print("Warning: gpt-4-turbo-preview may change over time. Estimating cost assuming gpt-4-0125-preview as it is the most recent version of gpt-4-turbo-preview.")
            return _estimate_cost(text, model="gpt-4-0125-preview")
        
        elif(model == "gpt-4-vision-preview"):
            print("Warning: gpt-4-vision-preview may change over time. Estimating cost assuming gpt-4-1106-vision-preview as it is the most recent version of gpt-4-1106-vision-preview.")
            return _estimate_cost(text, model="gpt-4-1106-vision-preview")
        
        elif(model == "gpt-4o"):
            print("Warning: gpt-4o may change over time. Estimating cost assuming gpt-4o-2024-05-13 as it is the most recent version of gpt-4o.")
            return _estimate_cost(text, model="gpt-4o-2024-05-13")
        
        elif(model == "gpt-3.5-turbo-0613"):
            print("Warning: gpt-3.5-turbo-0613 is considered depreciated by OpenAI as of November 6, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-3.5-turbo-0125.")
            return _estimate_cost(text, model=model, price_case=1)

        elif(model == "gpt-3.5-turbo-0301"):
            print("Warning: gpt-3.5-turbo-0301 is considered depreciated by OpenAI as of June 13, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-3.5-turbo-0125 unless you are specifically trying to break the filter.")
            return _estimate_cost(text, model=model, price_case=1)
        
        elif(model == "gpt-3.5-turbo-1106"):
            print("Warning: gpt-3.5-turbo-1106 is outdated, consider switching to gpt-3.5-turbo-0125.")
            return _estimate_cost(text, model=model, price_case=2)
        
        elif(model == "gpt-3.5-turbo-0125"):
            return _estimate_cost(text, model=model, price_case=7)
            
        elif(model == "gpt-3.5-turbo-16k-0613"):
            print("Warning: gpt-3.5-turbo-16k-0613 is considered depreciated by OpenAI as of November 6, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-3.5-turbo-1106.")
            return _estimate_cost(text, model=model, price_case=3)
        
        elif(model == "gpt-4-1106-preview"):
            return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "gpt-4-0125-preview"):
            return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "gpt-4-1106-vision-preview"):
            return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "gpt-4-turbo-2024-04-09"):
            return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "gpt-4-0314"):
            print("Warning: gpt-4-0314 is considered depreciated by OpenAI as of June 13, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-4-0613.")
            return _estimate_cost(text, model=model, price_case=5)
        
        elif(model == "gpt-4-0613"):
            return _estimate_cost(text, model=model, price_case=5)
                
        elif(model == "gpt-4-32k-0314"):
            print("Warning: gpt-4-32k-0314 is considered depreciated by OpenAI as of June 13, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-4-32k-0613.")
            return _estimate_cost(text, model=model, price_case=6)
        
        elif(model == "gpt-4-32k-0613"):
            return _estimate_cost(text, model=model, price_case=6)
        
        elif(model == "gpt-4o-2024-05-13"):
            return _estimate_cost(text, model=model, price_case=10)
        
        elif(model == "gemini-pro"):
            print(f"Warning: gemini-pro may change over time. Estimating cost assuming gemini-1.0-pro-001 as it is the most recent version of gemini-1.0-pro.")
            return _estimate_cost(text, model="gemini-1.0-pro-001", price_case=8)
        
        elif(model == "gemini-pro-vision"):
            print("Warning: gemini-pro-vision may change over time. Estimating cost assuming gemini-1.0-pro-vision-001 as it is the most recent version of gemini-1.0-pro-vision.")
            return _estimate_cost(text, model="gemini-1.0-pro-vision-001", price_case=8)
        
       ## elif(model == "gemini-ultra"):
    ##        return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "gemini-1.0-pro"):
            print(f"Warning: gemini-1.0-pro may change over time. Estimating cost assuming gemini-1.0-pro-001 as it is the most recent version of gemini-1.0-pro.")
            return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "gemini-1.0-pro-vision"):
            print("Warning: gemini-1.0-pro-vision may change over time. Estimating cost assuming gemini-1.0-pro-vision-001 as it is the most recent version of gemini-1.0-pro-vision.")
            return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "gemini-1.0-pro-latest"):
            print(f"Warning: gemini-1.0-pro-latest may change over time. Estimating cost assuming gemini-1.0-pro-001 as it is the most recent version of gemini-1.0-pro.")
            return _estimate_cost(text, model="gemini-1.0-pro-001", price_case=8)
        
        elif(model == "gemini-1.0-pro-vision-latest"):
            print("Warning: gemini-1.0-pro-vision-latest may change over time. Estimating cost assuming gemini-1.0-pro-vision-001 as it is the most recent version of gemini-1.0-pro-vision.")
            return _estimate_cost(text, model="gemini-1.0-pro-vision-001", price_case=8)
        
        elif(model == "gemini-1.5-pro"):
            return _estimate_cost(text, model=model, price_case=9)
        
        elif(model == "gemini-1.5-flash"):
            return _estimate_cost(text, model=model, price_case=9)

        elif(model == "gemini-1.5-pro-latest"):
            print("Warning: gemini-1.5-pro-latest may change over time. Estimating cost assuming gemini-1.5-pro as it is the most recent version of gemini-1.5-pro.")
            return _estimate_cost(text, model="gemini-1.5-pro", price_case=9)
        
        elif(model == "gemini-1.5-flash-latest"):
            print("Warning: gemini-1.5-flash-latest may change over time. Estimating cost assuming gemini-1.5-flash as it is the most recent version of gemini-1.5-flash.")
            return _estimate_cost(text, model="gemini-1.5-flash", price_case=9)
        
  ##      elif(model == "gemini-1.0-ultra-latest"):
      ##      return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "gemini-1.0-pro-001"):
            return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "gemini-1.0-pro-vision-001"):
            return _estimate_cost(text, model=model, price_case=8)
        
        elif(model == "claude-3-opus-20240229"):
            return _estimate_cost(text, model=model, price_case=13)
        
        elif(model == "claude-3-sonnet-20240229"):
            return _estimate_cost(text, model=model, price_case=12)
        
        elif(model == "claude-3-haiku-20240307"):
            return _estimate_cost(text, model=model, price_case=11)
        
    else:

        _cost_details = MODEL_COSTS.get(model)

        if(not _cost_details):
            raise ValueError(f"Cost details not found for model: {model}.")

        ## break down the text into a string than into tokens
        text = ''.join(text)

        model_types = {
            "openai": ALLOWED_OPENAI_MODELS,
            "gemini": ALLOWED_GEMINI_MODELS,
            "anthropic": ALLOWED_ANTHROPIC_MODELS
        }
        
        _LLM_TYPE = next((model_type for model_type, allowed_models in model_types.items() if model in allowed_models))

        if(_LLM_TYPE == "openai"):
            _encoding = tiktoken.encoding_for_model(model)
            _num_tokens = len(_encoding.encode(text))

        elif(_LLM_TYPE == "gemini"):
            ## no local option, and it seems to rate limit too lol, so we'll do it openai style

                _encoding = tiktoken.encoding_for_model("gpt-4-turbo-0125")
                _num_tokens = len(_encoding.encode(text))

        else:
            ## literally no way exists to get the number of tokens for anthropic, so we'll just use the gpt-4-turbo-0125 model as a stand-in
            _encoding = tiktoken.encoding_for_model("gpt-4-turbo-0125")
            _num_tokens = len(_encoding.encode(text))
            pass

        _input_cost = _cost_details["_input_cost"]
        _output_cost = _cost_details["_output_cost"]

        _min_cost_for_input = (_num_tokens / 1000) * _input_cost
        _min_cost_for_output = (_num_tokens / 1000) * _output_cost
        _min_cost = _min_cost_for_input + _min_cost_for_output

        return _num_tokens, _min_cost, model
    
    ## type checker doesn't like the chance of None being returned, so we raise an exception here if it gets to this point, which it shouldn't
    raise Exception("An unknown error occurred while calculating the minimum cost of translation.")

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Costs & Models are determined and updated manually, listed in USD. Updated by Bikatr7 as of 2024-05-19
## https://platform.openai.com/docs/models/overview
ALLOWED_OPENAI_MODELS  = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-1106-vision-preview",
    "gpt-4o",
    "gpt-4o-2024-05-13"
]

VALID_JSON_OPENAI_MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-1106-vision-preview",
    "gpt-4o-2024-05-13",
    "gpt-4o"    
]

## https://ai.google.dev/models/gemini
ALLOWED_GEMINI_MODELS = [
    "gemini-1.0-pro-001",
    "gemini-1.0-pro",
    "gemini-1.0-pro-latest",
    "gemini-1.0-pro-vision-001",
    "gemini-1.0-pro-vision",
    "gemini-1.0-pro-vision-latest",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
  ##  "gemini-1.0-ultra-latest",
    "gemini-pro",
    "gemini-pro-vision",
  ##  "gemini-ultra"
]

VALID_JSON_GEMINI_MODELS = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]

## https://docs.anthropic.com/en/docs/models-overview
ALLOWED_ANTHROPIC_MODELS = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

VALID_JSON_ANTHROPIC_MODELS = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

MODEL_COSTS = {
    # Grouping GPT-3.5 models together
    "gpt-3.5-turbo-0125": {"price_case": 7, "_input_cost": 0.0005, "_output_cost": 0.0015},
    "gpt-3.5-turbo-0301": {"price_case": 1, "_input_cost": 0.0015, "_output_cost": 0.0020},
    "gpt-3.5-turbo-0613": {"price_case": 1, "_input_cost": 0.0015, "_output_cost": 0.0020},
    "gpt-3.5-turbo-1106": {"price_case": 2, "_input_cost": 0.0010, "_output_cost": 0.0020},
    "gpt-3.5-turbo-16k-0613": {"price_case": 3, "_input_cost": 0.0030, "_output_cost": 0.0040},
    
    ## Grouping GPT-4 models by their capabilities and versions
    "gpt-4-0314": {"price_case": 5, "_input_cost": 0.03, "_output_cost": 0.06},
    "gpt-4-0613": {"price_case": 5, "_input_cost": 0.03, "_output_cost": 0.06},
    "gpt-4-32k-0314": {"price_case": 6, "_input_cost": 0.06, "_output_cost": 0.012},
    "gpt-4-32k-0613": {"price_case": 6, "_input_cost": 0.06, "_output_cost": 0.012},
    "gpt-4-1106-preview": {"price_case": 8, "_input_cost": 0.01, "_output_cost": 0.03},
    "gpt-4-0125-preview": {"price_case": 8, "_input_cost": 0.01, "_output_cost": 0.03},
    "gpt-4-1106-vision-preview": {"price_case": 8, "_input_cost": 0.01, "_output_cost": 0.03},
    "gpt-4-turbo-2024-04-09": {"price_case": 8, "_input_cost": 0.01, "_output_cost": 0.03},

    ## Grouping GPT-4o models together
    "gpt-4o-2024-05-13": {"price_case": 10, "_input_cost": 0.005, "_output_cost": 0.015},
    
    ## Grouping Gemini models together
    "gemini-1.0-pro-001": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-1.0-pro-vision-001": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-1.0-pro": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-1.0-pro-vision": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-1.0-pro-latest": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-1.0-pro-vision-latest": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-1.5-pro-latest": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
 ##   "gemini-1.0-ultra-latest": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-pro": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-pro-vision": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
 ##   "gemini-ultra": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0}
    "gemini-1.5-pro": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-1.5-flash-latest": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},
    "gemini-1.5-flash": {"price_case": 9, "_input_cost": 0.0, "_output_cost": 0.0},

    ## grouping anthropic models together
    "claude-3-haiku-20240307": {"price_case": 11, "_input_cost": 0.00025, "_output_cost": 0.00125},
    "claude-3-sonnet-20240229": {"price_case": 12, "_input_cost": 0.003, "_output_cost": 0.015},
    "claude-3-opus-20240229": {"price_case": 13, "_input_cost": 0.015, "_output_cost": 0.075}

}

MODEL_MAX_TOKENS = {

    ## openai models
    "gpt-4o-2024-05-13": {"max_input_tokens": 128000, "max_output_tokens": 4096},
    "gpt-4-turbo-2024-04-09": {"max_input_tokens": 128000, "max_output_tokens": 4096},
    "gpt-4-0125-preview": {"max_input_tokens": 128000, "max_output_tokens": 4096},
    "gpt-4-1106-preview": {"max_input_tokens": 128000, "max_output_tokens": 4096},
    "gpt-4-1106-vision-preview": {"max_input_tokens": 128000, "max_output_tokens": 4096},
    "gpt-4-0613": {"max_input_tokens": 8192, "max_output_tokens": 4096},
    "gpt-4-32k-0613": {"max_input_tokens": 32768, "max_output_tokens": 4096},
    "gpt-3.5-turbo-0125": {"max_input_tokens": 16385, "max_output_tokens": 4096},
    "gpt-3.5-turbo-1106": {"max_input_tokens": 16385, "max_output_tokens": 4096},
    "gpt-3.5-turbo-0613": {"max_input_tokens": 4096, "max_output_tokens": 4096},
    "gpt-3.5-turbo-16k-0613": {"max_input_tokens": 16385, "max_output_tokens": 4096},

    ## gemini models
    "gemini-1.5-pro": {"max_input_tokens": 1048576, "max_output_tokens": 8192},
    "gemini-1.5-flash": {"max_input_tokens": 1048576, "max_output_tokens": 8192},
    "gemini-1.0-pro-001": {"max_input_tokens": 12288, "max_output_tokens": 4096},
    "gemini-1.0-pro-vision-001": {"max_input_tokens": 12288, "max_output_tokens": 4096},

    ## anthropic models
    "claude-3-opus-20240229": {"max_input_tokens": 200000, "max_output_tokens": 4096},
    "claude-3-sonnet-20240229": {"max_input_tokens": 200000, "max_output_tokens": 4096},
    "claude-3-haiku-20240307": {"max_input_tokens": 200000, "max_output_tokens": 4096}

}