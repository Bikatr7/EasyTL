from easytl import EasyTL

import asyncio
import os
import time

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

    if(json_value is None):
        google_tl_key_path = "tests/google_translate_key.json"
        
        logging_directory = "tests/"
        gemini_time_delay = 5

    assert deepl_api_key is not None, "DEEPL_API_KEY environment variable must be set"
    assert gemini_api_key is not None, "GEMINI_API_KEY environment variable must be set"
    assert openai_api_key is not None, "OPENAI_API_KEY environment variable must be set"
    assert google_tl_key_path is not None, "GOOGLE_TRANSLATE_SERVICE_KEY_VALUE environment variable must be set"

    EasyTL.set_credentials("deepl", deepl_api_key)
    EasyTL.set_credentials("gemini", gemini_api_key)
    EasyTL.set_credentials("openai", openai_api_key)
    EasyTL.set_credentials("google translate", google_tl_key_path)

    return gemini_time_delay, logging_directory

##-------------------start-of-main()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

async def main():

    gemini_time_delay, logging_directory = setup_preconditions()

    print("------------------------------------------------Deepl------------------------------------------------")

    print("------------------------------------------------Text response------------------------------------------------")

    print(EasyTL.deepl_translate("Hello, world!", target_lang="DE", logging_directory=logging_directory))
    print(await EasyTL.deepl_translate_async("Hello, world!", target_lang="DE", logging_directory=logging_directory))
    
    print(EasyTL.deepl_translate("Hello, world!", target_lang="DE", response_type="raw", logging_directory=logging_directory).text) # type: ignore
    result = await EasyTL.deepl_translate_async("Hello, world!", target_lang="DE", response_type="raw", logging_directory=logging_directory)

    print(result.text) # type: ignore

    print("------------------------------------------------Raw response------------------------------------------------")

    results = EasyTL.deepl_translate(text=["Hello, world!", "Goodbye, world!"], target_lang="DE", response_type="raw", logging_directory=logging_directory)
    async_results = await EasyTL.deepl_translate_async(text=["Hello, world!", "Goodbye, world!"], target_lang="DE", response_type="raw", logging_directory=logging_directory)

    for result in results: # type: ignore
        print(result.text) # type: ignore

    for result in async_results: # type: ignore
        print(result.text)

    print("------------------------------------------------Cost calculation------------------------------------------------")

    characters, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="deepl", model=None, translation_instructions=None)

    print(f"Characters: {characters}, Cost: {cost}, Model: {model}")

    print("------------------------------------------------Google Translate------------------------------------------------")

    print("------------------------------------------------Text response------------------------------------------------")

    print(EasyTL.googletl_translate("Hello, world!", target_lang="de", logging_directory=logging_directory))
    print(await EasyTL.googletl_translate_async("Hello, world!", target_lang="de", logging_directory=logging_directory))

    print(EasyTL.googletl_translate("Hello, world!", target_lang="de", response_type="raw", logging_directory=logging_directory)["translatedText"]) # type: ignore
    result = await EasyTL.googletl_translate_async("Hello, world!", target_lang="de", response_type="raw", logging_directory=logging_directory)

    print(result["translatedText"]) # type: ignore

    print("------------------------------------------------Raw response------------------------------------------------")

    results = EasyTL.googletl_translate(text=["Hello, world!", "Goodbye, world!"], target_lang="de", response_type="raw", logging_directory=logging_directory)
    async_results = await EasyTL.googletl_translate_async(text=["Hello, world!", "Goodbye, world!"], target_lang="de", response_type="raw", logging_directory=logging_directory)

    for result in results: # type: ignore
        print(result["translatedText"]) # type: ignore

    for result in async_results: # type: ignore
        print(result["translatedText"]) # type: ignore

    print("------------------------------------------------Cost calculation------------------------------------------------")

    characters, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="google translate", model=None, translation_instructions=None)

    print(f"Characters: {characters}, Cost: {cost}, Model: {model}")

    print("------------------------------------------------Gemini------------------------------------------------")

    print("-----------------------------------------------Text response-----------------------------------------------")

    print(EasyTL.gemini_translate("Hello, world!", translation_instructions="Translate this to German.", logging_directory=logging_directory))
    print(await EasyTL.gemini_translate_async("Hello, world!", translation_instructions="Translate this to German.", logging_directory=logging_directory))

    print(EasyTL.gemini_translate("Hello, world!", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory).text) # type: ignore
    result = await EasyTL.gemini_translate_async("Hello, world!", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory)

    print(result.text) # type: ignore

    print("-----------------------------------------------Raw response-----------------------------------------------")

    results = EasyTL.gemini_translate(text=["Hello, world!", "Goodbye, world!"], translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory)
    async_results = await EasyTL.gemini_translate_async(text=["Hello, world!", "Goodbye, world!"], translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory, translation_delay=5)

    for result in results: # type: ignore
        print(result.text) # type: ignore

    for result in async_results: # type: ignore
        print(result.text)

    print("-----------------------------------------------JSON response-----------------------------------------------")

    print(EasyTL.gemini_translate("Hello, world!", model="gemini-1.5-pro-latest", translation_instructions="Translate this to German. Format the response as JSON.", response_type="json", logging_directory=logging_directory))
    
    time.sleep(gemini_time_delay)
    
    print(await EasyTL.gemini_translate_async("Hello, world!", model="gemini-1.5-pro-latest", translation_instructions="Format the response as JSON parseable string. It should have 2 keys, one for input titled input, and one called output, which is the translation.", response_type="json", logging_directory=logging_directory))

    print("------------------------------------------------Cost calculation------------------------------------------------")

    tokens, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="gemini", model="gemini-pro", translation_instructions="Translate this to German.")

    print(f"Tokens: {tokens}, Cost: {cost}, Model: {model}")

    print("------------------------------------------------OpenAI------------------------------------------------")

    print("-----------------------------------------------Text response-----------------------------------------------")

    print(EasyTL.openai_translate("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", logging_directory=logging_directory))
    print(await EasyTL.openai_translate_async("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", logging_directory=logging_directory))

    print(EasyTL.openai_translate("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory).choices[0].message.content) # type: ignore
    result = await EasyTL.openai_translate_async("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory)

    print(result.choices[0].message.content) # type: ignore

    print("-----------------------------------------------Raw response-----------------------------------------------")

    results = EasyTL.openai_translate(text=["Hello, world!", "Goodbye, world!"], model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory)
    async_results = await EasyTL.openai_translate_async(text=["Hello, world!", "Goodbye, world!"], model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory=logging_directory)

    for result in results: # type: ignore
        print(result.choices[0].message.content) # type: ignore

    for result in async_results: # type: ignore
        print(result.choices[0].message.content ) # type: ignore

    print("-----------------------------------------------JSON response-----------------------------------------------")

    print(EasyTL.openai_translate("Hello, world!", model="gpt-3.5-turbo-0125", translation_instructions="Translate this to German. Format the response as JSON parseable string.", response_type="json", logging_directory=logging_directory))
    print(await EasyTL.openai_translate_async("Hello, world!", model="gpt-3.5-turbo-0125", translation_instructions="Translate this to German. Format the response as JSON parseable string. It should have 2 keys, one for input titled input, and one called output, which is the translation.", response_type="json", logging_directory=logging_directory))

    print("------------------------------------------------Cost calculation------------------------------------------------")

    tokens, cost, model = EasyTL.calculate_cost(text="Hello, world!", service="openai", model="gpt-3.5-turbo", translation_instructions="Translate this to German.")

    print(f"Tokens: {tokens}, Cost: {cost}, Model: {model}")

##-------------------end-of-main()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if(__name__ == "__main__"):
    asyncio.run(main())