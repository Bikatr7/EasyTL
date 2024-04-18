from easytl import EasyTL

import asyncio


def read_api_key(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

async def main():

    deepl_api_key = read_api_key('tests/deepl.txt')
    gemini_api_key = read_api_key('tests/gemini.txt')
    openai_api_key = read_api_key('tests/openai.txt')

    EasyTL.set_api_key("deepl", deepl_api_key)
    EasyTL.set_api_key("gemini", gemini_api_key)
    EasyTL.set_api_key("openai", openai_api_key)


    print(EasyTL.deepl_translate("Hello, world!", target_lang="DE", logging_directory="C:/Users/Tetra/Desktop/"))
    print(await EasyTL.deepl_translate_async("Hello, world!", target_lang="DE", logging_directory="C:/Users/Tetra/Desktop/"))
    
    print(EasyTL.deepl_translate("Hello, world!", target_lang="DE", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/").text) # type: ignore
    result = await EasyTL.deepl_translate_async("Hello, world!", target_lang="DE", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/")

    print(result.text) # type: ignore

    results = EasyTL.deepl_translate(text=["Hello, world!", "Goodbye, world!"], target_lang="DE", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/")
    async_results = await EasyTL.deepl_translate_async(text=["Hello, world!", "Goodbye, world!"], target_lang="DE", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/")

    for result in results: # type: ignore
        print(result.text) # type: ignore

    for result in async_results: # type: ignore
        print(result.text)

    print(EasyTL.gemini_translate("Hello, world!", translation_instructions="Translate this to German.", logging_directory="C:/Users/Tetra/Desktop/"))
    print(await EasyTL.gemini_translate_async("Hello, world!", translation_instructions="Translate this to German.", logging_directory="C:/Users/Tetra/Desktop/"))

    print(EasyTL.gemini_translate("Hello, world!", translation_instructions="Translate this to German.", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/").text) # type: ignore
    result = await EasyTL.gemini_translate_async("Hello, world!", translation_instructions="Translate this to German.", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/")

    print(result.text) # type: ignore

    results = EasyTL.gemini_translate(text=["Hello, world!", "Goodbye, world!"], translation_instructions="Translate this to German.", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/")
    async_results = await EasyTL.gemini_translate_async(text=["Hello, world!", "Goodbye, world!"], translation_instructions="Translate this to German.", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/")

    for result in results: # type: ignore
        print(result.text) # type: ignore

    for result in async_results: # type: ignore
        print(result.text)


    print(EasyTL.openai_translate("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", logging_directory="C:/Users/Tetra/Desktop/"))
    print(await EasyTL.openai_translate_async("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", logging_directory="C:/Users/Tetra/Desktop/"))

    print(EasyTL.openai_translate("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/").choices[0].message.content) # type: ignore
    result = await EasyTL.openai_translate_async("Hello, world!", model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/")

    print(result.choices[0].message.content) # type: ignore

    results = EasyTL.openai_translate(text=["Hello, world!", "Goodbye, world!"], model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/")
    async_results = await EasyTL.openai_translate_async(text=["Hello, world!", "Goodbye, world!"], model="gpt-3.5-turbo", translation_instructions="Translate this to German.", response_type="raw", logging_directory="C:/Users/Tetra/Desktop/")

    for result in results: # type: ignore
        print(result.choices[0].message.content) # type: ignore

    for result in async_results: # type: ignore
        print(result.choices[0].message.content ) # type: ignore



if(__name__ == "__main__"):
    asyncio.run(main())