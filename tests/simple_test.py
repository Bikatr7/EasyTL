from easytl import EasyTL

import asyncio
import os
import time
import logging

import backoff

from easytl.exceptions import DeepLException, GoogleAPIError, OpenAIError, AnthropicError

##-------------------start-of-read_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_api_key(filename):
    try:
        with open(filename, 'r') as file:
            return file.read().strip()
        
    except:
        pass

def setup_preconditions():

    gemini_time_delay = 30

    deepl_api_key = os.environ.get('DEEPL_API_KEY')
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
    json_value = os.environ.get('GOOGLE_TRANSLATE_SERVICE_KEY_VALUE')

    if(json_value is not None):

        with open("json_value.txt", "w") as file:
            file.write(json_value)

        google_tl_key_path = "json_value.txt"

    logging_directory = os.getenv('LOGGING_DIRECTORY', '/tmp/')

    if(deepl_api_key is None):
        deepl_api_key = read_api_key("tests/deepl.txt")
    if(gemini_api_key is None):
        gemini_api_key = read_api_key("tests/gemini.txt")
    if(openai_api_key is None):
        openai_api_key = read_api_key("tests/openai.txt")
    if(anthropic_api_key is None):
        anthropic_api_key = read_api_key("tests/anthropic.txt")

    if(json_value is None):
        google_tl_key_path = "tests/google_translate_key.json"
        
        logging_directory = "tests/"
        gemini_time_delay = 5

    assert deepl_api_key is not None, "DEEPL_API_KEY environment variable must be set"
    assert gemini_api_key is not None, "GEMINI_API_KEY environment variable must be set"
    assert openai_api_key is not None, "OPENAI_API_KEY environment variable must be set"
    assert google_tl_key_path is not None, "GOOGLE_TRANSLATE_SERVICE_KEY_VALUE environment variable must be set"
    assert anthropic_api_key is not None, "ANTHROPIC_API_KEY environment variable must be set"

    EasyTL.set_credentials("deepl", deepl_api_key)
    EasyTL.set_credentials("gemini", gemini_api_key)
    EasyTL.set_credentials("openai", openai_api_key)
    EasyTL.set_credentials("google translate", google_tl_key_path)
    EasyTL.set_credentials("anthropic", anthropic_api_key)

    return gemini_time_delay, logging_directory

##-------------------start-of-main()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

async def main():

    gemini_time_delay, logging_directory = setup_preconditions()

    decorator = backoff.on_exception(backoff.expo, exception=(DeepLException, GoogleAPIError, OpenAIError, AnthropicError), logger=logging.getLogger())

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

    print(EasyTL.anthropic_translate("Hello, world!", translation_instructions="Translate this to German.", response_type="json", logging_directory=logging_directory,decorator=decorator, response_schema=schema))
    print(await EasyTL.anthropic_translate_async("Hello, world!", translation_instructions="Translate this to German.", response_type="json", logging_directory=logging_directory,decorator=decorator, response_schema=schema))

    tokens, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="anthropic", model="claude-3-haiku-20240307", translation_instructions="Translate this to German.")

    print(f"Tokens: {tokens}, Cost: {cost}, Model: {model}")
    
 ##   print(EasyTL.openai_translate("Hello, world!", model="gpt-3.5-turbo-0125", translation_instructions="Translate this to German in json format.", response_type="json", logging_directory=logging_directory,decorator=decorator))

##    print(EasyTL.deepl_translate("Hello, world!", target_lang="DE", response_type="text", logging_directory=logging_directory,decorator=decorator))

 ##   print(EasyTL.googletl_translate("Hello, world!", target_lang="de", response_type="text", logging_directory=logging_directory,decorator=decorator))

  ##  print(EasyTL.gemini_translate("Hello, world!", model="gemini-pro", response_type="text", logging_directory=logging_directory,decorator=decorator))

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
