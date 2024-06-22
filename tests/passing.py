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

    ## schema for gemini & anthropic
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

    return gemini_time_delay, logging_directory, schema, azure_region

##-------------------start-of-main()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

async def main():

    gemini_time_delay, logging_directory, schema, azure_region = setup_preconditions()

    ## probably self explanatory from this point on

    decorator = backoff.on_exception(backoff.expo, exception=(DeepLException, GoogleAPIError, OpenAIError, AnthropicAPIError), logger=logging.getLogger())

    print("------------------------------------------------Deepl------------------------------------------------")

    print("------------------------------------------------Text response------------------------------------------------")

    print(EasyTL.deepl_translate(text="Hello, world!", target_lang="DE", logging_directory=logging_directory, decorator=decorator))
    print(await EasyTL.deepl_translate_async(text="Hello, world!", target_lang="DE", logging_directory=logging_directory,decorator=decorator))

    print("------------------------------------------------Raw response------------------------------------------------")

    results = EasyTL.deepl_translate(text=["Hello, world!", "Goodbye, world!"], target_lang="DE", response_type="raw", logging_directory=logging_directory, decorator=decorator)
    async_results = await EasyTL.deepl_translate_async(text=["Hello, world!", "Goodbye, world!"], target_lang="DE", response_type="raw", logging_directory=logging_directory, decorator=decorator)

    for result in results: # type: ignore
        print(result.text) # type: ignore

    for result in async_results: # type: ignore
        print(result.text)

    print("------------------------------------------------Cost calculation------------------------------------------------")

    characters, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="deepl", model=None, translation_instructions=None)

    print(f"Characters: {characters}, Cost: {cost}, Model: {model}")

    print("------------------------------------------------Google Translate------------------------------------------------")

    print("------------------------------------------------Text response------------------------------------------------")

    print(EasyTL.googletl_translate("Hello, world!", target_lang="de", logging_directory=logging_directory, decorator=decorator))
    print(await EasyTL.googletl_translate_async("Hello, world!", target_lang="de", logging_directory=logging_directory, decorator=decorator))

    print("------------------------------------------------Raw response------------------------------------------------")

    results = EasyTL.googletl_translate(text=["Hello, world!", "Goodbye, world!"], target_lang="de", response_type="raw", logging_directory=logging_directory, decorator=decorator)
    async_results = await EasyTL.googletl_translate_async(text=["Hello, world!", "Goodbye, world!"], target_lang="de", response_type="raw", logging_directory=logging_directory, decorator=decorator)

    for result in results: # type: ignore
        print(result["translatedText"]) # type: ignore

    for result in async_results: # type: ignore
        print(result["translatedText"]) # type: ignore

    print("------------------------------------------------Cost calculation------------------------------------------------")

    characters, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="google translate", model=None, translation_instructions=None)

    print(f"Characters: {characters}, Cost: {cost}, Model: {model}")

    print("------------------------------------------------Gemini------------------------------------------------")

    print("-----------------------------------------------Text response-----------------------------------------------")

    print(EasyTL.gemini_translate("Hello, world!", translation_instructions="Translate this to German.", logging_directory=logging_directory, decorator=decorator))
    print(await EasyTL.gemini_translate_async("Hello, world!", translation_instructions="Translate this to German.", logging_directory=logging_directory, decorator=decorator))

    print("-----------------------------------------------Raw response-----------------------------------------------")

    results = EasyTL.gemini_translate(text=["Hello, world!", "Goodbye, world!"], translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory,decorator=decorator)
    async_results = await EasyTL.gemini_translate_async(text=["Hello, world!", "Goodbye, world!"], translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory, translation_delay=5,decorator=decorator)

    for result in results: # type: ignore
        print(result.text) # type: ignore

    for result in async_results: # type: ignore
        print(result.text)

    print("-----------------------------------------------JSON response-----------------------------------------------")

    print(EasyTL.gemini_translate("Hello, world!", model="gemini-1.5-pro-latest", translation_instructions="Translate this to German. Format the response as JSON parseable string must have 2 keys, one for input titled input, and one called output, which is the translation.", response_type="json", logging_directory=logging_directory,decorator=decorator, response_schema=schema))
    
    time.sleep(gemini_time_delay)
    
    print(await EasyTL.gemini_translate_async("Hello, world!", model="gemini-1.5-pro-latest", translation_instructions="Format the response as JSON parseable string. It should have 2 keys, one for input titled input, and one called output, which is the translation.", response_type="json", logging_directory=logging_directory,decorator=decorator))

    print("------------------------------------------------Cost calculation------------------------------------------------")

    tokens, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="gemini", model="gemini-pro", translation_instructions="Translate this to German.")

    print(f"Tokens: {tokens}, Cost: {cost}, Model: {model}")

    print("------------------------------------------------OpenAI------------------------------------------------")

    print("-----------------------------------------------Text response-----------------------------------------------")

    print(EasyTL.openai_translate("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", logging_directory=logging_directory, decorator=decorator))
    print(await EasyTL.openai_translate_async("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", logging_directory=logging_directory, decorator=decorator))

    print("-----------------------------------------------Raw response-----------------------------------------------")

    results = EasyTL.openai_translate(text=["Hello, world!", "Goodbye, world!"], model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory,decorator=decorator)
    async_results = await EasyTL.openai_translate_async(text=["Hello, world!", "Goodbye, world!"], model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory,decorator=decorator)

    for result in results: # type: ignore
        print(result.choices[0].message.content) # type: ignore

    for result in async_results: # type: ignore
        print(result.choices[0].message.content ) # type: ignore

    print("-----------------------------------------------JSON response-----------------------------------------------")

    print(EasyTL.openai_translate("Hello, world!", model="gpt-3.5-turbo-0125", translation_instructions="Translate this to German. Format the response as JSON parseable string.", response_type="json", logging_directory=logging_directory,decorator=decorator))
    print(await EasyTL.openai_translate_async("Hello, world!", model="gpt-3.5-turbo-0125", translation_instructions="Translate this to German. Format the response as JSON parseable string. It should have 2 keys, one for input titled input, and one called output, which is the translation.", response_type="json", logging_directory=logging_directory,decorator=decorator))

    print("------------------------------------------------Cost calculation------------------------------------------------")

    tokens, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="openai", model="gpt-3.5-turbo", translation_instructions="Translate this to German.")

    print(f"Tokens: {tokens}, Cost: {cost}, Model: {model}")

    print("------------------------------------------------Anthropic------------------------------------------------")

    print("-----------------------------------------------Text response-----------------------------------------------")

    print(EasyTL.anthropic_translate("Hello, world!", model="claude-3-haiku-20240307", translation_instructions="Translate this to German.", logging_directory=logging_directory, decorator=decorator))
    print(await EasyTL.anthropic_translate_async("Hello, world!", model="claude-3-haiku-20240307", translation_instructions="Translate this to German.", logging_directory=logging_directory, decorator=decorator))

    print("-----------------------------------------------Raw response-----------------------------------------------")

    results = EasyTL.anthropic_translate(text=["Hello, world!", "Goodbye, world!"], model="claude-3-haiku-20240307", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory,decorator=decorator)
    async_results = await EasyTL.anthropic_translate_async(text=["Hello, world!", "Goodbye, world!"], model="claude-3-haiku-20240307", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory,decorator=decorator)

    for result in results: # type: ignore
        print(result.content[0].text) # type: ignore

    for result in async_results: # type: ignore
        print(result.content[0].text) # type: ignore

    print("-----------------------------------------------JSON response-----------------------------------------------")

    print(EasyTL.anthropic_translate("Hello, world!", model="claude-3-haiku-20240307", translation_instructions="Translate this to German. Format the response as JSON parseable string.", response_type="json", logging_directory=logging_directory,decorator=decorator, response_schema=schema))
    print(await EasyTL.anthropic_translate_async("Hello, world!", model="claude-3-haiku-20240307", translation_instructions="Translate this to German. Format the response as JSON parseable string.", response_type="json", logging_directory=logging_directory,decorator=decorator, response_schema=schema))

    print("------------------------------------------------Cost calculation------------------------------------------------")

    tokens, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="anthropic", model="claude-3-haiku-20240307", translation_instructions="Translate this to German.")

    print(f"Tokens: {tokens}, Cost: {cost}, Model: {model}")

    print("------------------------------------------------Azure------------------------------------------------")
    
    print("-----------------------------------------------Text response-----------------------------------------------")

    print(EasyTL.azure_translate("Hello, world!", target_lang="de", logging_directory=logging_directory, decorator=decorator, azure_region=azure_region))
    print(await EasyTL.azure_translate_async("Hello, world!", target_lang="de", logging_directory=logging_directory, decorator=decorator, azure_region=azure_region))

    print("-----------------------------------------------JSON response-----------------------------------------------")

    print(EasyTL.azure_translate("Hello, world!", target_lang="de", response_type="json", logging_directory=logging_directory,decorator=decorator, azure_region=azure_region))
    print(await EasyTL.azure_translate_async("Hello, world!", target_lang="de", response_type="json", logging_directory=logging_directory,decorator=decorator, azure_region=azure_region))

    print("------------------------------------------------Cost calculation------------------------------------------------")

    characters, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="azure", model=None, translation_instructions=None)

    print(f"Characters: {characters}, Cost: {cost}, Model: {model}")   

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
