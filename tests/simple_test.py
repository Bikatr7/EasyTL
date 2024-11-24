## Copyright 2024 Kaden Bilyeu (Bikatr7) (https://github.com/Bikatr7), Alejandro Mata (https://github.com/alemalvarez)
## EasyTL (https://github.com/Bikatr7/EasyTL)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## built-in libraries
import asyncio
import os
import time
import logging

## third-party libraries
from pydantic import BaseModel

import backoff

## custom modules
from easytl import EasyTL

from easytl.exceptions import DeepLException, GoogleAPIError, OpenAIError, AnthropicAPIError

##-------------------start-of-read_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_api_key(filename):

    try:
        with open(filename, 'r') as file:
            return file.read().strip()
    except:
        pass

##-------------------start-of-setup_preconditions()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def setup_preconditions():

    ## default values, assuming github actions environment, because i ain't paying shit
    gemini_time_delay = 65 

    deepl_api_key = os.environ.get('DEEPL_API_KEY')
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    json_value = os.environ.get('GOOGLE_TRANSLATE_SERVICE_KEY_VALUE')
    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
    azure_api_key = os.environ.get('AZURE_API_KEY')
    azure_region = os.environ.get('AZURE_REGION')

    ## this dumps the json value to a file as that's what the google translate service expects
    if(json_value is not None):

        with open("json_value.txt", "w") as file:
            file.write(json_value)

        google_tl_key_path = "json_value.txt"

    ## the following will replace the keys with local test keys if the environment variables are not set (not a github actions environment)
    logging_directory = os.getenv('LOGGING_DIRECTORY', '/tmp/')

    if(deepl_api_key is None):
        deepl_api_key = read_api_key("tests/deepl.txt")
    if(gemini_api_key is None):
        gemini_api_key = read_api_key("tests/gemini.txt")
    if(openai_api_key is None):
        openai_api_key = read_api_key("tests/openai.txt")
    if(anthropic_api_key is None):
        anthropic_api_key = read_api_key("tests/anthropic.txt")
    if(azure_api_key is None):
        azure_api_key = read_api_key("tests/azure.txt")
    if(azure_region is None):
        azure_region = read_api_key("tests/azure_region.txt")

    ## last json failure clarifies that this is not a github actions environment
    if(json_value is None):
        google_tl_key_path = "tests/google_translate_key.json"
        
        logging_directory = "tests/"
        gemini_time_delay = 5

    ## if any of these trigger, something is wrong (not in local test environment or github actions environment)
    assert deepl_api_key is not None, "DEEPL_API_KEY environment variable must be set"
    assert gemini_api_key is not None, "GEMINI_API_KEY environment variable must be set"
    assert openai_api_key is not None, "OPENAI_API_KEY environment variable must be set"
    assert anthropic_api_key is not None, "ANTHROPIC_API_KEY environment variable must be set"
    assert azure_api_key is not None, "AZURE_API_KEY environment variable must be set"
    #assert azure_region is not None, "AZURE_REGION environment variable must be set" 
    # we can set a default for the region
    if(azure_region is None):
        azure_region = "westus"
        print(f"Using default Azure region: {azure_region}")
        
    assert google_tl_key_path is not None, "GOOGLE_TRANSLATE_SERVICE_KEY_VALUE environment variable must be set"

    ## set the credentials for the services
    EasyTL.set_credentials("deepl", deepl_api_key)
    EasyTL.set_credentials("gemini", gemini_api_key)
    EasyTL.set_credentials("openai", openai_api_key)
    EasyTL.set_credentials("anthropic", anthropic_api_key)
    EasyTL.set_credentials("google translate", google_tl_key_path)
    EasyTL.set_credentials("azure", azure_api_key)

    ## non_openai_schema for gemini & anthropic
    non_openai_schema = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The text you were given to translate"
            },
            "output": {
                "type": "string",
                "description": "The translated text"
            }
        },
        "required": ["input", "output"]
    }

    openai_schema = {
        "name": "convert_to_json",
        "schema": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The text you were given to translate"
                },
                "output": {
                    "type": "string",
                    "description": "The translated text"
                }
            },
            "required": ["input", "output"]
        }
    }

    return gemini_time_delay, logging_directory, non_openai_schema, openai_schema, azure_region

##-------------------start-of-main()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class ConvertToJson(BaseModel):
    input:str
    output:str

async def main():

    gemini_time_delay, logging_directory, non_openai_schema, openai_schema, azure_region = setup_preconditions()


    decorator = backoff.on_exception(backoff.expo, exception=(DeepLException, GoogleAPIError, OpenAIError, AnthropicAPIError), logger=logging.getLogger())

    schema = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "The text you were given to translate"
            },
            "output": {
                "type": "string",
                "description": "The translated text"
            }
        },
        "required": ["input", "output"]
    }

    # print("Testing Gemini Streaming...")

    # stream_response = EasyTL.gemini_translate("Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
    #                                           model="gemini-1.5-flash", 
    #                                           translation_instructions="Translate this to German. Take your time and translate word by word.", 
    #                                           stream=True,
    #                                           decorator=decorator)
    
    # for chunk in stream_response: # type: ignore
    #     if hasattr(chunk, 'text') and chunk.text is not None: # type: ignore
    #         print(chunk.text, end="", flush=True) # type: ignore
    #         time.sleep(0.1)
    # print()

    # print("Testing Gemini Async Streaming...")

    # async_stream_response = await EasyTL.gemini_translate_async("Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
    #                                                             model="gemini-1.5-flash", 
    #                                                             translation_instructions="Translate this to German. Take your time and translate word by word.",
    #                                                             stream=True,
    #                                                             decorator=decorator)
    
    # async for chunk in async_stream_response: # type: ignore
    #     if hasattr(chunk, 'text') and chunk.text is not None: # type: ignore
    #         print(chunk.text, end="", flush=True)
    #         await asyncio.sleep(0.1)
    # print()

    print("Testing Anthropic Streaming...")

    stream_response = EasyTL.anthropic_translate("Hello, world! This is a longer message to better demonstrate streaming capabilities.",
                                                translation_instructions="Translate this to German.", 
                                                stream=True, 
                                                decorator=decorator)

    for event in stream_response: # type: ignore
        if event.type == "message_start":
            print("Translation started...")
        elif event.type == "content_block_start":
            print("Content block started...")
        elif event.type == "content_block_delta":
            if hasattr(event.delta, 'text'):
                print(event.delta.text, end="", flush=True)
                time.sleep(0.1)
        elif event.type == "content_block_stop":
            print("\nContent block finished...")
        elif event.type == "message_delta":
            if hasattr(event.delta, 'text'):
                print(event.delta.text, end="", flush=True)
                time.sleep(0.1)
        elif event.type == "message_stop":
            print("\nTranslation completed.")
    print()

    print("Testing Anthropic Async Streaming...")

    async_stream_response = await EasyTL.anthropic_translate_async("Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
                                                                   translation_instructions="Translate this to German.", 
                                                                   stream=True, 
                                                                   decorator=decorator)

    async for event in async_stream_response: # type: ignore
        if event.type == "message_start":
            print("Translation started...")
        elif event.type == "content_block_start":
            print("Content block started...")
        elif event.type == "content_block_delta":
            if hasattr(event.delta, 'text'):
                print(event.delta.text, end="", flush=True)
                await asyncio.sleep(0.1)
        elif event.type == "content_block_stop":
            print("\nContent block finished...")
        elif event.type == "message_delta":
            if hasattr(event.delta, 'text'):
                print(event.delta.text, end="", flush=True)
                await asyncio.sleep(0.1)
        elif event.type == "message_stop":
            print("\nTranslation completed.")
    print()

    # print(EasyTL.anthropic_translate("Hello, world!", translation_instructions="Translate this to German.", response_type="json", logging_directory=logging_directory,decorator=decorator, response_schema=schema))

    ## print(EasyTL.azure_translate("Hello, world!", target_lang="de", logging_directory=logging_directory, decorator=decorator, azure_region=azure_region))


    ## print("Testing OpenAI Streaming...")

    ## print("\nSync Streaming:")
    ## stream_response = EasyTL.openai_translate("Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
    ##                                         model="gpt-4", 
    ##                                         translation_instructions="Translate this to German. Take your time and translate word by word.", 
    ##                                         stream=True,
    ##                                         decorator=decorator)
    
    ## for chunk in stream_response: # type: ignore
    ##     if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
    ##         print(chunk.choices[0].delta.content, end="", flush=True)
    ##         time.sleep(0.1)
    ## print()

    ## print("\nAsync Streaming:")
    ## async_stream_response = await EasyTL.openai_translate_async(
    ##     "Hello, world! This is a longer message that we'll see stream in real time. Each word should appear one by one.", 
    ##     model="gpt-4", 
    ##     translation_instructions="Translate this to German. Take your time and translate word by word.",
    ##     stream=True,
    ##     decorator=decorator
    ## )
    
    ## async for chunk in async_stream_response: # type: ignore
    ##     if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
    ##         print(chunk.choices[0].delta.content, end="", flush=True)
    ##         await asyncio.sleep(0.1)
    ## print()

    ## print("Testing Non-Streaming...")

    ## print(EasyTL.openai_translate("Hello, world! This is a longer message to better demonstrate streaming capabilities.", 
    ##                               model="gpt-4", 
    ##                               translation_instructions="Translate this to German. Take your time and translate word by word.", 
    ##                               stream=False,
    ##                               decorator=decorator))

##-------------------end-of-main()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if(__name__ == "__main__"):
    ## setup logging
    logging.basicConfig(level=logging.DEBUG, 
                        filename='passing.log',
                        filemode='w', 
                        format='[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    asyncio.run(main())
