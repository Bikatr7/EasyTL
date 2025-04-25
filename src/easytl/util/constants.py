## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## Costs & Models are determined and updated manually, listed in USD. Updated by Bikatr7 as of 2025-04-25
## https://platform.openai.com/docs/models/overview

## needs aliases as well as specific model dates
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
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1",
    "o1-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    "gpt-4.5-preview",
    "gpt-4.5-preview-2025-02-27",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano",
    "gpt-4.1-nano-2025-04-14",
    "o4-mini",
    "o4-mini-2025-04-16"
    
]

## specific model dates only
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
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.5-preview-2025-02-27",
    "o3-mini-2025-01-31",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "o4-mini-2025-04-16"
]

## specific model dates only
VALID_STRUCTURED_OUTPUT_OPENAI_MODELS = [
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "o1",
    "o1-2024-09-12",
    "o3-mini-2025-01-31",
    "gpt-4.5-preview-2025-02-27",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "o4-mini-2025-04-16"
]

## https://ai.google.dev/models/gemini
## specific model dates and aliases
ALLOWED_GEMINI_MODELS = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-pro",
    "gemini-pro-vision",
    "gemini-2.0-flash-live",
    "gemini-2.0-flash-live-001",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-flash-preview-04-17"
]

## specific model dates only
VALID_JSON_GEMINI_MODELS = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-live",
    "gemini-2.0-flash-live-001",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-flash-preview-04-17"
]

## https://docs.anthropic.com/en/docs/models-overview
## specific model dates and aliases
ALLOWED_ANTHROPIC_MODELS = [
    "claude-3-5-haiku-latest",
    "claude-3-7-sonnet-latest",
    "claude-3-opus-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022", 
    "claude-3-7-sonnet-20250219",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022"
]

VALID_JSON_ANTHROPIC_MODELS = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219"
]

## per thousand, not per million
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
    "gpt-4o-2024-08-06": {"price_case": 17, "_input_cost": 0.0025, "_output_cost": 0.010},
    "gpt-4o-2024-11-20": {"price_case": 17, "_input_cost": 0.0025, "_output_cost": 0.010},
    "gpt-4o-audio-preview-2024-12-17": {"price_case": 17, "_input_cost": 0.0025, "_output_cost": 0.010},
    "gpt-4o-realtime-preview-2024-12-17": {"price_case": 26, "_input_cost": 0.005, "_output_cost": 0.020},
    "gpt-4o-search-preview-2025-03-11": {"price_case": 17, "_input_cost": 0.0025, "_output_cost": 0.010},
    "gpt-4o-mini-2024-07-18": {"price_case": 16, "_input_cost": 0.000150, "_output_cost": 0.000600},
    "gpt-4o-mini-audio-preview-2024-12-17": {"price_case": 16, "_input_cost": 0.00015, "_output_cost": 0.0006},
    "gpt-4o-mini-realtime-preview-2024-12-17": {"price_case": 27, "_input_cost": 0.0006, "_output_cost": 0.0024},
    "gpt-4o-mini-search-preview-2025-03-11": {"price_case": 16, "_input_cost": 0.00015, "_output_cost": 0.0006},

    ## Grouping o1 models together
    "o1-preview-2024-09-12": {"price_case": 18, "_input_cost": 0.015, "_output_cost": 0.060},
    "o1-2024-09-12": {"price_case": 18, "_input_cost": 0.015, "_output_cost": 0.060},
    "o1-2024-12-17": {"price_case": 18, "_input_cost": 0.015, "_output_cost": 0.060},
    "o1-pro-2025-03-19": {"price_case": 28, "_input_cost": 0.150, "_output_cost": 0.600},
    "o1-mini-2024-09-12": {"price_case": 20, "_input_cost": 0.0011, "_output_cost": 0.0044},

    ## Grouping o3 models together
    "o3-2025-04-16": {"price_case": 29, "_input_cost": 0.010, "_output_cost": 0.040},
    "o3-mini-2025-01-31": {"price_case": 20, "_input_cost": 0.0011, "_output_cost": 0.0044},

    ## Grouping o4 models together
    "o4-mini-2025-04-16": {"price_case": 20, "_input_cost": 0.0011, "_output_cost": 0.0044},

    ## Grouping gpt-4.5-preview-2025-02-27 models together
    "gpt-4.5-preview-2025-02-27": {"price_case": 21, "_input_cost": 0.0750, "_output_cost": 0.1500},

    ## Grouping computer use models together
    "computer-use-preview-2025-03-11": {"price_case": 19, "_input_cost": 0.003, "_output_cost": 0.012},

    ## Grouping image models together
    "gpt-image-1": {"price_case": 30, "_input_cost": 0.005, "_output_cost": 0.000},
    
    ## Grouping Gemini models together
    "gemini-pro": {"price_case": 9, "_input_cost": 0.0005, "_output_cost": 0.0015},
    "gemini-pro-vision": {"price_case": 9, "_input_cost": 0.0005, "_output_cost": 0.0015},

    "gemini-1.5-pro-latest": {"price_case": 14, "_input_cost": 0.00125, "_output_cost": 0.005},
    "gemini-1.5-pro": {"price_case": 14, "_input_cost": 0.00125, "_output_cost": 0.005},
    "gemini-1.5-pro-001": {"price_case": 14, "_input_cost": 0.00125, "_output_cost": 0.005},
    "gemini-1.5-pro-002": {"price_case": 14, "_input_cost": 0.00125, "_output_cost": 0.005},

    "gemini-1.5-flash-latest": {"price_case": 16, "_input_cost": 0.00015, "_output_cost": 0.0006},
    "gemini-1.5-flash": {"price_case": 16, "_input_cost": 0.00015, "_output_cost": 0.0006},
    "gemini-1.5-flash-001": {"price_case": 16, "_input_cost": 0.00015, "_output_cost": 0.0006},
    "gemini-1.5-flash-002": {"price_case": 16, "_input_cost": 0.00015, "_output_cost": 0.0006},
    "gemini-1.5-flash-8b": {"price_case": 33, "_input_cost": 0.0000375, "_output_cost": 0.00015},

    "gemini-2.0-flash-latest": {"price_case": 25, "_input_cost": 0.0001, "_output_cost": 0.0004},

    ## grouping anthropic models together
    "claude-3-haiku-20240307": {"price_case": 22, "_input_cost": 0.0008, "_output_cost": 0.004},
    "claude-3-5-haiku-20241022": {"price_case": 11, "_input_cost": 0.00025, "_output_cost": 0.00125},
    "claude-3-sonnet-20240229": {"price_case": 12, "_input_cost": 0.003, "_output_cost": 0.015},
    "claude-3-5-sonnet-20240620": {"price_case": 12, "_input_cost": 0.003, "_output_cost": 0.015},
    "claude-3-5-sonnet-20241022": {"price_case": 12, "_input_cost": 0.003, "_output_cost": 0.015},
    "claude-3-7-sonnet-20250219": {"price_case": 12, "_input_cost": 0.003, "_output_cost": 0.015},
    "claude-3-opus-20240229": {"price_case": 13, "_input_cost": 0.015, "_output_cost": 0.075}

}

## i am not fucking updating this shit
MODEL_MAX_TOKENS = {

    ## openai models
    "o1-preview-2024-09-12": {"max_input_tokens": 128000, "max_output_tokens": 32768},
    "o1-mini-2024-09-12": {"max_input_tokens": 128000, "max_output_tokens": 65536},
    "o1-2024-09-12": {"max_input_tokens": 200000, "max_output_tokens": 100000},
    "gpt-4o-mini-2024-07-18": {"max_input_tokens": 128000, "max_output_tokens": 16384},
    "gpt-4o-2024-05-13": {"max_input_tokens": 128000, "max_output_tokens": 4096},
    "gpt-4o-2024-08-06": {"max_input_tokens": 128000, "max_output_tokens": 16384},
    "gpt-4o-2024-11-20": {"max_input_tokens": 128000, "max_output_tokens": 16384},
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
    "gpt-4.5-preview-2025-02-27": {"max_input_tokens": 128000, "max_output_tokens": 16384},
    "o3-mini-2025-01-31": {"max_input_tokens": 200000, "max_output_tokens": 100000},

    ## gemini models
    "gemini-1.5-pro": {"max_input_tokens": 1048576, "max_output_tokens": 8192},
    "gemini-1.5-flash": {"max_input_tokens": 1048576, "max_output_tokens": 8192},
    "gemini-2.0-flash-latest": {"max_input_tokens": 1048576, "max_output_tokens": 8192},

    ## anthropic models
    "claude-3-opus-20240229": {"max_input_tokens": 200000, "max_output_tokens": 4096},
    "claude-3-sonnet-20240229": {"max_input_tokens": 200000, "max_output_tokens": 4096},
    "claude-3-5-sonnet-20240620": {"max_input_tokens": 200000, "max_output_tokens": 8192},
    "claude-3-5-sonnet-20241022": {"max_input_tokens": 200000, "max_output_tokens": 8192},
    "claude-3-7-sonnet-20250219": {"max_input_tokens": 200000, "max_output_tokens": 8192},
    "claude-3-haiku-20240307": {"max_input_tokens": 200000, "max_output_tokens": 4096},
    "claude-3-5-haiku-20241022": {"max_input_tokens": 200000, "max_output_tokens": 8192}

}