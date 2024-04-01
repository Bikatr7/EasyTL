## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

from typing import Iterable

def convert_iterable_to_str(iterable:Iterable) -> str:
    return "".join(map(str, iterable))

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
