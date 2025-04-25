## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing

## third-party libraries
import tiktoken

## custom modules
from .constants import ALLOWED_OPENAI_MODELS, ALLOWED_GEMINI_MODELS, ALLOWED_ANTHROPIC_MODELS, MODEL_COSTS

from ..classes import NotGiven, NOT_GIVEN


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
        "openai_base_url": {"_type": str, "constraints": lambda x: isinstance(x, str) or x is None or x is NOT_GIVEN},
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

    assert model in ALLOWED_OPENAI_MODELS + ALLOWED_GEMINI_MODELS + ALLOWED_ANTHROPIC_MODELS, f"""EasyTL does not support : {model}, if you believe this is an error, please report it to the EasyTL GitHub repository (https://github.com/Bikatr7/EasyTL)."""

    ## default models are first, then the rest are sorted by price case
    if(price_case is None):

        if(model == "gpt-3.5-turbo"):
            print("Warning: gpt-3.5-turbo may change over time. Estimating cost assuming gpt-3.5-turbo-0125 as it is the most recent version of gpt-3.5-turbo.")
            return _estimate_cost(text, model="gpt-3.5-turbo-0125", price_case=7)
        
        elif(model == "gpt-3.5-turbo-16k"):
            print("Warning: gpt-3.5-turbo-16k may change over time. Estimating cost assuming gpt-3.5-turbo-16k-0613 as it is the most recent version of gpt-3.5-turbo-16k.")
            return _estimate_cost(text, model="gpt-3.5-turbo-16k-0613", price_case=3)
        
        elif(model == "gpt-4"):
            print("Warning: gpt-4 may change over time. Estimating cost assuming gpt-4-0613 as it is the most recent version of gpt-4.")
            return _estimate_cost(text, model="gpt-4-0613", price_case=5)
        
        elif(model == "gpt-4-32k"):
            print("Warning: gpt-4-32k may change over time. Estimating cost assuming gpt-4-32k-0613 as it is the most recent version of gpt-4-32k.")
            return _estimate_cost(text, model="gpt-4-32k-0613", price_case=6)
        
        elif(model == "gpt-4-turbo"):
            print("Warning: gpt-4-turbo may change over time. Estimating cost assuming gpt-4-turbo-2024-04-09 as it is the most recent version of gpt-4-turbo.")
            return _estimate_cost(text, model="gpt-4-turbo-2024-04-09", price_case=8)
                
        elif(model == "gpt-4-turbo-preview"):
            print("Warning: gpt-4-turbo-preview may change over time. Estimating cost assuming gpt-4-0125-preview as it is the most recent version of gpt-4-turbo-preview.")
            return _estimate_cost(text, model="gpt-4-0125-preview", price_case=8)
        
        elif(model == "gpt-4-vision-preview"):
            print("Warning: gpt-4-vision-preview may change over time. Estimating cost assuming gpt-4-1106-vision-preview as it is the most recent version of gpt-4-1106-vision-preview.")
            return _estimate_cost(text, model="gpt-4-1106-vision-preview", price_case=8)
        
        elif(model == "gpt-4o"):
            print("Warning: gpt-4o may change over time. Estimating cost assuming gpt-4o-2024-11-20 as it is the most recent version of gpt-4o.")
            return _estimate_cost(text, model="gpt-4o-2024-11-20", price_case=17)
        
        elif(model == "gpt-4o-mini"):
            print("Warning: gpt-4o-mini may change over time. Estimating cost assuming gpt-4o-mini-2024-07-18 as it is the most recent version of gpt-4o-mini.")
            return _estimate_cost(text, model="gpt-4o-mini-2024-07-18", price_case=16)

        elif(model == "gpt-4o-audio-preview"):
            print("Warning: gpt-4o-audio-preview may change over time. Estimating cost assuming gpt-4o-audio-preview-2024-12-17 as it is the most recent version.")
            return _estimate_cost(text, model="gpt-4o-audio-preview-2024-12-17", price_case=17)

        elif(model == "gpt-4o-realtime-preview"):
            print("Warning: gpt-4o-realtime-preview may change over time. Estimating cost assuming gpt-4o-realtime-preview-2024-12-17 as it is the most recent version.")
            return _estimate_cost(text, model="gpt-4o-realtime-preview-2024-12-17", price_case=26)

        elif(model == "gpt-4o-search-preview"):
             print("Warning: gpt-4o-search-preview may change over time. Estimating cost assuming gpt-4o-search-preview-2025-03-11 as it is the most recent version.")
             return _estimate_cost(text, model="gpt-4o-search-preview-2025-03-11", price_case=17)

        elif(model == "gpt-4o-mini-audio-preview"):
            print("Warning: gpt-4o-mini-audio-preview may change over time. Estimating cost assuming gpt-4o-mini-audio-preview-2024-12-17 as it is the most recent version.")
            return _estimate_cost(text, model="gpt-4o-mini-audio-preview-2024-12-17", price_case=16)

        elif(model == "gpt-4o-mini-realtime-preview"):
            print("Warning: gpt-4o-mini-realtime-preview may change over time. Estimating cost assuming gpt-4o-mini-realtime-preview-2024-12-17 as it is the most recent version.")
            return _estimate_cost(text, model="gpt-4o-mini-realtime-preview-2024-12-17", price_case=27)

        elif(model == "gpt-4o-mini-search-preview"):
            print("Warning: gpt-4o-mini-search-preview may change over time. Estimating cost assuming gpt-4o-mini-search-preview-2025-03-11 as it is the most recent version.")
            return _estimate_cost(text, model="gpt-4o-mini-search-preview-2025-03-11", price_case=16)

        elif(model == "gpt-4.1"):
            print("Warning: gpt-4.1 may change over time. Estimating cost assuming gpt-4.1-2025-04-14 as it is the most recent version.")
            return _estimate_cost(text, model="gpt-4.1-2025-04-14", price_case=23)

        elif(model == "gpt-4.1-mini"):
            print("Warning: gpt-4.1-mini may change over time. Estimating cost assuming gpt-4.1-mini-2025-04-14 as it is the most recent version.")
            return _estimate_cost(text, model="gpt-4.1-mini-2025-04-14", price_case=24)

        elif(model == "gpt-4.1-nano"):
            print("Warning: gpt-4.1-nano may change over time. Estimating cost assuming gpt-4.1-nano-2025-04-14 as it is the most recent version.")
            return _estimate_cost(text, model="gpt-4.1-nano-2025-04-14", price_case=25)

        elif(model == "gpt-4.5-preview"):
            print("Warning: gpt-4.5-preview may change over time. Estimating cost assuming gpt-4.5-preview-2025-02-27 as it is the most recent version.")
            return _estimate_cost(text, model="gpt-4.5-preview-2025-02-27", price_case=21)
        
        elif(model == "o1-preview"):
            print("Warning: o1-preview may change over time. Estimating cost assuming o1-preview-2024-09-12 as it is the most recent version of o1-preview.")
            return _estimate_cost(text, model="o1-preview-2024-09-12", price_case=18)
        
        elif(model == "o1-mini"):
            print("Warning: o1-mini may change over time. Estimating cost assuming o1-mini-2024-09-12 as it is the most recent version of o1-mini.")
            return _estimate_cost(text, model="o1-mini-2024-09-12", price_case=20)
        
        elif(model == "o1"):
            print("Warning: o1 may change over time. Estimating cost assuming o1-2024-12-17 as it is the most recent version of o1.")
            return _estimate_cost(text, model="o1-2024-12-17", price_case=18)

        elif(model == "o1-pro"):
            print("Warning: o1-pro may change over time. Estimating cost assuming o1-pro-2025-03-19 as it is the most recent version.")
            return _estimate_cost(text, model="o1-pro-2025-03-19", price_case=28)

        elif(model == "o3"):
            print("Warning: o3 may change over time. Estimating cost assuming o3-2025-04-16 as it is the most recent version.")
            return _estimate_cost(text, model="o3-2025-04-16", price_case=29)
        
        elif(model == "o3-mini"):
            print("Warning: o3-mini may change over time. Estimating cost assuming o3-mini-2025-01-31 as it is the most recent version.")
            return _estimate_cost(text, model="o3-mini-2025-01-31", price_case=20)

        elif(model == "o4-mini"):
            print("Warning: o4-mini may change over time. Estimating cost assuming o4-mini-2025-04-16 as it is the most recent version.")
            return _estimate_cost(text, model="o4-mini-2025-04-16", price_case=20)

        elif(model == "computer-use-preview"):
            print("Warning: computer-use-preview may change over time. Estimating cost assuming computer-use-preview-2025-03-11 as it is the most recent version.")
            return _estimate_cost(text, model="computer-use-preview-2025-03-11", price_case=19)

        elif(model == "gpt-image-1"):
            ## No specific date, assume it's stable for now
            return _estimate_cost(text, model="gpt-image-1", price_case=30)
        
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

        elif(model == "gpt-4.1-2025-04-14"):
            return _estimate_cost(text, model=model, price_case=23)

        elif(model == "gpt-4.1-mini-2025-04-14"):
            return _estimate_cost(text, model=model, price_case=24)

        elif(model == "gpt-4.1-nano-2025-04-14"):
            return _estimate_cost(text, model=model, price_case=25)
        
        elif(model == "gpt-4o-mini-2024-07-18"):
            return _estimate_cost(text, model=model, price_case=16)

        elif(model == "gpt-4o-mini-audio-preview-2024-12-17"):
            return _estimate_cost(text, model=model, price_case=16)

        elif(model == "gpt-4o-mini-realtime-preview-2024-12-17"):
            return _estimate_cost(text, model=model, price_case=27)

        elif(model == "gpt-4o-mini-search-preview-2025-03-11"):
             return _estimate_cost(text, model=model, price_case=16)

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
        
        elif(model == "gpt-4o-2024-08-06"):
            return _estimate_cost(text, model=model, price_case=17)
        
        elif(model == "gpt-4o-2024-11-20"):
            return _estimate_cost(text, model=model, price_case=17)

        elif(model == "gpt-4o-audio-preview-2024-12-17"):
            return _estimate_cost(text, model=model, price_case=17)

        elif(model == "gpt-4o-realtime-preview-2024-12-17"):
            return _estimate_cost(text, model=model, price_case=26)

        elif(model == "gpt-4o-search-preview-2025-03-11"):
            return _estimate_cost(text, model=model, price_case=17)
        
        elif(model == "o1-preview-2024-09-12"):
            return _estimate_cost(text, model=model, price_case=18)
        
        elif(model == "o1-mini-2024-09-12"):
            return _estimate_cost(text, model=model, price_case=20)
        
        elif(model == "o1-2024-12-17"):
            return _estimate_cost(text, model=model, price_case=18)

        elif(model == "o1-pro-2025-03-19"):
            return _estimate_cost(text, model=model, price_case=28)

        elif(model == "o3-2025-04-16"):
            return _estimate_cost(text, model=model, price_case=29)
        
        elif(model == "o4-mini-2025-04-16"):
            return _estimate_cost(text, model=model, price_case=20)

        elif(model == "computer-use-preview-2025-03-11"):
            return _estimate_cost(text, model=model, price_case=19)
                
        elif(model == "gemini-pro"):
             print(f"Warning: gemini-pro alias used. Routing to gemini-1.5-pro-002.")
             return _estimate_cost(text, model="gemini-1.5-pro-002", price_case=14)

        elif(model == "gemini-1.5-pro"):
            print("Warning: gemini-1.5-pro may change over time. Using latest known version gemini-1.5-pro-002.")
            return _estimate_cost(text, model="gemini-1.5-pro-002", price_case=14)
        
        elif(model == "gemini-1.5-pro-latest"):
            print("Warning: gemini-1.5-pro-latest may change over time. Estimating cost assuming gemini-1.5-pro-002 as it is the most recent version of gemini-1.5-pro.")
            return _estimate_cost(text, model="gemini-1.5-pro-002", price_case=14)
        
        elif(model == "gemini-1.5-flash"):
            print("Warning: gemini-1.5-flash may change over time. Using latest known version gemini-1.5-flash-002.")
            return _estimate_cost(text, model="gemini-1.5-flash-002", price_case=16)

        elif(model == "gemini-1.5-flash-latest"):
            print("Warning: gemini-1.5-flash-latest may change over time. Estimating cost assuming gemini-1.5-flash-002 as it is the most recent version of gemini-1.5-flash.")
            return _estimate_cost(text, model="gemini-1.5-flash-002", price_case=16)
        
        elif(model == "gemini-1.5-flash-8b"):
            ## Assuming this has a specific date or is stable for now
            return _estimate_cost(text, model="gemini-1.5-flash-8b", price_case=33)

        elif(model == "gemini-2.0-flash-latest"):
            ## Assuming this points to the latest stable/dated version
            return _estimate_cost(text, model="gemini-2.0-flash-latest", price_case=25)

        elif(model == "gemini-2.0-flash"):
             ## Assuming this points to the latest stable/dated version, same as latest
            return _estimate_cost(text, model="gemini-2.0-flash", price_case=25)

        elif(model == "gemini-2.0-flash-lite"):
            ## Assuming this has a specific date or is stable for now
            return _estimate_cost(text, model="gemini-2.0-flash-lite", price_case=32)

        elif(model == "gemini-2.5-pro-preview-03-25"):
            return _estimate_cost(text, model=model, price_case=31)

        elif(model == "gemini-2.5-flash-preview-04-17"):
             return _estimate_cost(text, model=model, price_case=16)
                
        elif(model == "claude-3-opus-20240229"):
            return _estimate_cost(text, model=model, price_case=13)
        
        elif(model == "claude-3-sonnet-20240229"):
            return _estimate_cost(text, model=model, price_case=12)
        
        elif(model == "claude-3-5-sonnet-20240620"):
            return _estimate_cost(text, model=model, price_case=12)
        
        elif(model == "claude-3-haiku-20240307"):
            return _estimate_cost(text, model=model, price_case=22)
        
        elif(model == "claude-3-5-haiku-20241022"):
            return _estimate_cost(text, model=model, price_case=11)
        
        elif(model == "claude-3-5-sonnet-20241022"):
            return _estimate_cost(text, model=model, price_case=12)
            
        elif(model == "claude-3-7-sonnet-20250219"):
            return _estimate_cost(text, model=model, price_case=12)
        
        elif(model == "gpt-4o-2024-11-20"):
            return _estimate_cost(text, model=model, price_case=17)
        
        elif(model == "o1-2024-09-12"):
            return _estimate_cost(text, model=model, price_case=18)
        
        elif(model == "o3-mini-2025-01-31"):
            return _estimate_cost(text, model=model, price_case=20)
        
        elif(model == "gpt-4.5-preview-2025-02-27"):
            return _estimate_cost(text, model=model, price_case=21)
        
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
            try:
                _encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                ## Fallback for newer models not yet in tiktoken
                print(f"Warning: Encoding for model {model} not found. Using cl100k_base as fallback.")
                _encoding = tiktoken.get_encoding("cl100k_base")
            _num_tokens = len(_encoding.encode(text))

        elif(_LLM_TYPE == "gemini"):
            ## no local option, and it seems to rate limit too lol, so we'll do it openai style
            try:
                _encoding = tiktoken.encoding_for_model("gpt-4-turbo-2024-04-09") ## Use a recent common model as proxy
            except KeyError:
                print("Warning: Encoding for fallback model gpt-4-turbo-2024-04-09 not found. Using cl100k_base.")
                _encoding = tiktoken.get_encoding("cl100k_base")
            _num_tokens = len(_encoding.encode(text))

        else: ## Anthropic
            ## literally no way exists to get the number of tokens for anthropic, so we'll just use the gpt-4-turbo-0125 model as a stand-in
            try:
                _encoding = tiktoken.encoding_for_model("gpt-4-turbo-2024-04-09") ## Use a recent common model as proxy
            except KeyError:
                 print("Warning: Encoding for fallback model gpt-4-turbo-2024-04-09 not found. Using cl100k_base.")
                 _encoding = tiktoken.get_encoding("cl100k_base")
            _num_tokens = len(_encoding.encode(text))
            

        _input_cost = _cost_details["_input_cost"]
        _output_cost = _cost_details["_output_cost"]

        _min_cost_for_input = (_num_tokens / 1000) * _input_cost
        _min_cost_for_output = (_num_tokens / 1000) * _output_cost
        _min_cost = _min_cost_for_input + _min_cost_for_output

        return _num_tokens, _min_cost, model
    
    ## type checker doesn't like the chance of None being returned, so we raise an exception here if it gets to this point, which it shouldn't
    raise Exception(f"Could not determine price case or details for model: {model}")

##-------------------start-of-_update_model_name()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def _update_model_name(model: str) -> str:

    """
    
    Updates the model name to the most recent version if an alias is provided.

    Parameters:
    model (string) : the model to update.

    Returns:
    model (string) : the updated model name (specific version) or the original name if not an alias.

    """

    model_updates = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k-0613", ## Note: Deprecated, but keeping mapping for now
        "gpt-4": "gpt-4-0613", ## Note: Deprecated, but keeping mapping for now
        "gpt-4-32k": "gpt-4-32k-0613", ## Note: Deprecated, but keeping mapping for now
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview": "gpt-4-0125-preview",
        "gpt-4-vision-preview": "gpt-4-1106-vision-preview",
        "gpt-4o": "gpt-4o-2024-11-20",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4o-audio-preview": "gpt-4o-audio-preview-2024-12-17",
        "gpt-4o-realtime-preview": "gpt-4o-realtime-preview-2024-12-17",
        "gpt-4o-search-preview": "gpt-4o-search-preview-2025-03-11",
        "gpt-4o-mini-audio-preview": "gpt-4o-mini-audio-preview-2024-12-17",
        "gpt-4o-mini-realtime-preview": "gpt-4o-mini-realtime-preview-2024-12-17",
        "gpt-4o-mini-search-preview": "gpt-4o-mini-search-preview-2025-03-11",
        "gpt-4.1": "gpt-4.1-2025-04-14",
        "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
        "gpt-4.5-preview": "gpt-4.5-preview-2025-02-27",
        "o1-preview": "o1-preview-2024-09-12",
        "o1-mini": "o1-mini-2024-09-12",
        "o1": "o1-2024-12-17",
        "o1-pro": "o1-pro-2025-03-19",
        "o3": "o3-2025-04-16",
        "o3-mini": "o3-mini-2025-01-31",
        "o4-mini": "o4-mini-2025-04-16",
        "computer-use-preview": "computer-use-preview-2025-03-11",
        ## Gemini Aliases (assuming 'latest' points to the most recent dated one, adjust if needed)
        "gemini-pro": "gemini-1.5-pro-002",
        "gemini-1.5-pro": "gemini-1.5-pro-002", ## Assuming latest known dated version
        "gemini-1.5-pro-latest": "gemini-1.5-pro-002",
        "gemini-1.5-flash": "gemini-1.5-flash-002", ## Assuming latest known dated version
        "gemini-1.5-flash-latest": "gemini-1.5-flash-002",
        "gemini-2.0-flash": "gemini-2.0-flash-latest", ## Assuming the non-latest maps to latest here
        ## Anthropic Aliases (Map 'latest' to the most recent specific version)
        "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
        "claude-3-7-sonnet-latest": "claude-3-7-sonnet-20250219",
        "claude-3-opus-latest": "claude-3-opus-20240229",
        ## No alias for claude-3-5-sonnet-latest as per the provided list in constants.py, assuming user specifies dated version directly
    }

    if(model in model_updates):
        return model_updates[model]

    return model
