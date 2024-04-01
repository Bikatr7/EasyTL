## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import typing

## custom modules
from easytl.exceptions import InvalidEasyTLSettings

##-------------------start-of-string_to_bool()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def string_to_bool(string:str) -> bool:

    return string.lower() in ['true', '1', 'yes', 'y', 't']

##-------------------start-of-convert_iterable_to_str()-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def convert_iterable_to_str(iterable:typing.Iterable) -> str:
    return "".join(map(str, iterable))

##-------------------start-of-validate_easytl_translation_settings()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def validate_easytl_translation_settings(settings:dict, type:typing.Literal["gemini","openai"]) -> None:

    """

    Validates the Kijiku Rules.json file.

    """

    openai_keys = [
        "openai_model",
        "openai_system_message",
        "openai_temperature",
        "openai_top_p",
        "openai_n",
        "openai_stream",
        "openai_stop",
        "openai_logit_bias",
        "openai_max_tokens",
        "openai_presence_penalty",
        "openai_frequency_penalty"
    ]

    gemini_keys = [
        "gemini_model",
        "gemini_temperature",
        "gemini_top_p",
        "gemini_top_k",
    ##    "gemini_candidate_count",
    ##    "gemini_stream",
  ##      "gemini_stop_sequences",
        "gemini_max_output_tokens"
    ]

    validation_rules = {

        "openai_model": lambda x: isinstance(x, str) and x in ALLOWED_OPENAI_MODELS,
        "openai_system_message": lambda x: x not in ["", "None", None],
        "openai_temperature": lambda x: isinstance(x, float) and 0 <= x <= 2,
        "openai_top_p": lambda x: isinstance(x, float) and 0 <= x <= 1,
        "openai_max_tokens": lambda x: x is None or isinstance(x, int) and x > 0,
        "openai_presence_penalty": lambda x: isinstance(x, float) and -2 <= x <= 2,
        "gemini_model": lambda x: isinstance(x, str) and x in ALLOWED_GEMINI_MODELS,
        "gemini_prompt": lambda x: x not in ["", "None", None],
        "gemini_temperature": lambda x: isinstance(x, float) and 0 <= x <= 2,
        "gemini_top_p": lambda x: x is None or (isinstance(x, float) and 0 <= x <= 2),
        "gemini_top_k": lambda x: x is None or (isinstance(x, int) and x >= 0),
        "gemini_max_output_tokens": lambda x: x is None or isinstance(x, int),
##        "gemini_stop_sequences": lambda x: x is None or all(isinstance(i, str) for i in x)
    }
    
    try:

        ## assign to variables to reduce repetitive access    
        if(type == "openai"):


            ## ensure all keys are present
            assert all(key in settings for key in openai_keys)

            ## validate each key using the validation rules
            for key, validate in validation_rules.items():
                if(key in settings and not validate(settings[key])):
                    raise ValueError(f"Invalid value for {key}")
                
            ## force stop/logit_bias/stream into None
            settings["openai_stop"] = None
            settings["openai_logit_bias"] = None
            settings["openai_stream"] = False

            ## force n and candidate_count to 1
            settings["openai_n"] = 1

        elif(type == "gemini"):

            ## ensure all keys are present
            assert all(key in settings for key in gemini_keys)

            ## validate each key using the validation rules
            for key, validate in validation_rules.items():
                if (key in settings and not validate(settings[key])):
                    raise ValueError(f"Invalid value for {key}")
                
        ##    settings["gemini_stream"] = False
      ##      settings["gemini_candidate_count"] = 1
        
    except Exception as e:
        raise InvalidEasyTLSettings(f"Invalid settings, Due to: {str(e)}")

##-------------------start-of-convert_to_correct_type()-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def convert_to_correct_type(setting_name:str, initial_value:str) -> typing.Any:

    """

    Converts the input string to the correct type based on the setting name.

    Parameters:
    setting_name (str) : The name of the setting to convert.
    initial_value (str) : The initial value to convert.

    Returns:
    (typing.Any) : The converted value.

    """

    value = initial_value
    
    type_expectations = {
        "openai_model": {"type": str, "constraints": lambda x: x in ALLOWED_OPENAI_MODELS},
        "openai_system_message": {"type": str, "constraints": lambda x: x not in ["", "None", None]},
        "openai_temperature": {"type": float, "constraints": lambda x: 0 <= x <= 2},
        "openai_top_p": {"type": float, "constraints": lambda x: 0 <= x <= 2},
        "openai_n": {"type": int, "constraints": lambda x: x == 1},
        "openai_stream": {"type": bool, "constraints": lambda x: x is False},
        "openai_stop": {"type": None, "constraints": lambda x: x is None},
        "openai_logit_bias": {"type": None, "constraints": lambda x: x is None},
        "openai_max_tokens": {"type": int, "constraints": lambda x: x is None or isinstance(x, int)},
        "openai_presence_penalty": {"type": float, "constraints": lambda x: -2 <= x <= 2},
        "openai_frequency_penalty": {"type": float, "constraints": lambda x: -2 <= x <= 2},
        "gemini_model": {"type": str, "constraints": lambda x: x in ALLOWED_GEMINI_MODELS},
       ## "gemini_prompt": {"type": str, "constraints": lambda x: x not in ["", "None", None]},
        "gemini_temperature": {"type": float, "constraints": lambda x: 0 <= x <= 2},
        "gemini_top_p": {"type": float, "constraints": lambda x: x is None or (isinstance(x, float) and 0 <= x <= 2)},
        "gemini_top_k": {"type": int, "constraints": lambda x: x is None or x >= 0},
  ##      "gemini_candidate_count": {"type": int, "constraints": lambda x: x == 1},
      ##  "gemini_stream": {"type": bool, "constraints": lambda x: x is False},
   ## "gemini_stop_sequences": {"type": list, "constraints": lambda x: x is None or all(isinstance(i, str) for i in x)},
        "gemini_max_output_tokens": {"type": int, "constraints": lambda x: x is None or isinstance(x, int)},
    }

    if(setting_name not in type_expectations):
        raise ValueError("Invalid setting name")

    setting_info = type_expectations[setting_name]

    if("stream" in setting_name):
        value = string_to_bool(initial_value)

    elif(isinstance(initial_value, str) and initial_value.lower() in ["none","null"]):
        value = None

    if(setting_info["type"] is None):
        converted_value = None

    elif(setting_info["type"] == int) or (setting_info["type"] == float):

        if(value is None or value == ''):
            converted_value = None
            
        elif(setting_info["type"] == int):
            converted_value = int(value)

        else:
            converted_value = float(value)

    else:
        converted_value = setting_info["type"](value)

    if("constraints" in setting_info and not setting_info["constraints"](converted_value)):
        raise ValueError(f"{setting_name} out of range")

    return converted_value

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ALLOWED_OPENAI_MODELS  = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo-preview",
    "gpt-3.5-turbo-0301",
    "gpt-4-0314",
    "gpt-4-32k-0314",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview"
]

ALLOWED_GEMINI_MODELS = [
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-vision-001",
    "gemini-1.0-pro",
    "gemini-1.0-pro-vision",
    "gemini-1.0-pro-latest",
    "gemini-1.0-pro-vision-latest",
    "gemini-1.5-pro-latest",
    "gemini-1.0-ultra-latest",
    "gemini-pro",
    "gemini-pro-vision",
    "gemini-ultra"
]
