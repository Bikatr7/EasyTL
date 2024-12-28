---------------------------------------------------------------------------------------------------------------------------------------------------
**Table of Contents**

- [**Notes**](#notes)
- [**Quick Start**](#quick-start)
- [**Installation**](#installation)
- [**Features**](#features)
- [**API Usage**](#api-usage)
  - [Translating Text](#translating-text)
  - [Generic Translation Methods](#generic-translation-methods)
  - [Cost Calculation](#cost-calculation)
  - [Credentials Management](#credentials-management)
- [**License**](#license)
- [**Contact**](#contact)
- [**Contribution**](#contribution)
- [**Acknowledgements**](#acknowledgements)

--------------------------------------------------------------------------------------------------------------------------------------------------

## **Notes**<a name="notes"></a>

Seamless Multi-API Translation: Simplifying Language Barriers with DeepL, OpenAI, Gemini, Google Translate and More! 

EasyTL has a [Trello board](https://trello.com/b/Td555CoW/easytl) for tracking planned features and issues:

We've compiled a repository of examples and use cases for EasyTL at this [GitHub repository](https://github.com/Bikatr7/easytl-demo)

> [!TIP]
> You can find the full documentation [here](https://easytl.readthedocs.io/en/latest/index.html)! (work in progress)

EasyTL tends to update it's LLM internal dependencies at least once a month, this will result in at least one minor version update per month. This is to ensure that the package is up to date with the latest features and bug fixes from the LLM services.

Sometimes this will occur faster, if a critical bug is found or a new feature is added that is important to the package.

---------------------------------------------------------------------------------------------------------------------------------------------------
## **Quick Start**<a name="quick-start"></a>

To get started with EasyTL, install the package via pip:

```bash
pip install easytl
```

Then, you can translate text using by importing the global client.

For example, with DeepL:

```python
from easytl import EasyTL

## Set your API key
EasyTL.set_credentials("deepl", "YOUR_API_KEY")

## You can also validate your API keys; translation functions will do this automatically
is_valid, e = EasyTL.test_credentials("deepl")

translated_text = EasyTL.deepl_translate("私は日本語が話せます", "EN-US") ## Text to translate, language to translate to, only two "required" arguments but there are more optional arguments for additional functionality and other services.

print(translated_text) ## Output: "I can speak Japanese"
```

or with OpenAI:

```python
from easytl import EasyTL

import asyncio

async def main():

    ## Set your API key
    EasyTL.set_credentials("openai", "YOUR_API_KEY")

    ## Get's the raw response from the API, allowing you to access the full response object
    raw_response = await EasyTL.openai_translate_async("I can speak Japanese", model="gpt-4o", translation_instructions="Translate this text to Japanese.", response_type="raw") 

    print(raw_response.choices[0].message.content) ## Output: "私は日本語が話せます" or something similar

if(__name__ == "__main__"):
    asyncio.run(main())

```

---------------------------------------------------------------------------------------------------------------------------------------------------

## **Installation**<a name="installation"></a>

Python 3.10+

EasyTL can be installed using pip:

```bash
pip install easytl
```

This will install EasyTL along with its dependencies and requirements.

These are the dependencies/requirements that will be installed:
```bash
setuptools>=61.0
wheel
setuptools_scm>=6.0
tomli
google-generativeai>=0.8.3
deepl>=1.20.0
openai>=1.58.1
backoff>=2.2.1
tiktoken>=0.7.0
google-cloud-translate>=3.15.3
anthropic>=0.42.0
requests>=2.31.0
pydantic
```
---------------------------------------------------------------------------------------------------------------------------------------------------

## **Features**<a name="features"></a>

EasyTL offers seamless integration with several translation APIs, allowing users to easily switch between services based on their needs. Key features include:

- Support for multiple translation APIs including OpenAI, DeepL, Gemini, Google Translate, Microsoft Azure Translate, and Anthropic.
- Simple API key and credential management and validation.
- Cost estimation tools to help manage usage based on text length, translation instructions for LLMs, and translation services.
- Highly customizable translation options, with each API's original features and more. 
- Lots of optional arguments for additional functionality. Such as decorators, semaphores, and rate-limit delays.

---------------------------------------------------------------------------------------------------------------------------------------------------

## **API Usage**<a name="api-usage"></a>

### Translating Text

Translate functions can be broken down into two categories: LLM and non-LLM. LLM ones can take instructions, while non-LLM ones require a target language. 

`deepl_translate`, `googletl_translate`, and `azure_translate` are non-LLM functions, while `openai_translate`, `gemini_translate`, and `anthropic_translate` are LLM functions.

Each method accepts various parameters to customize the translation process, such as language, text format, and API-specific features like formality level or temperature. However these vary wildly between services, so it is recommended to check the documentation for each service for more information.

All services offer asynchronous translation methods that return a future object for concurrent processing. These methods are suffixed with `_async` and can be awaited to retrieve the translated text.

Instead of receiving the translated text directly, you can also use the `response_type` parameter to get the raw response object, specify a json response where available, or both.
  
  `text` - Default. Returns the translated text.

  `json` - Returns the response as a JSON object. Not all services support this.

  `raw` - Returns the raw response object from the API. This can be useful for accessing additional information or debugging.
  
  `raw_json` - Returns the raw response object with the text but with the response also a json object. Again, not all services support this.

### Generic Translation Methods

EasyTL has generic translation methods `translate` and `translate_async` that can be used to translate text with any of the supported services. These methods accept the text, service, and kwargs of the respective service as parameters.

### Cost Calculation

The `calculate_cost` method provides an estimate of the cost associated with translating a given text with specified settings for each supported service.

These are characters or tokens depending on the type of translate function used.

```python
num_characters, cost, model = EasyTL.calculate_cost("This has a lot of characters", "deepl")
```

or 

```python
num_tokens, cost, model = EasyTL.calculate_cost("This has a lot of tokens.", "openai", model="gpt-4", translation_instructions="Translate this text to Japanese.")
```

### Credentials Management

Credentials can be set and validated using `set_credentials` and `test_credentials` methods to ensure they are active and correct before submitting translation requests.

If you don't provide an api key, the package will attempt to read it from the environment variables. The format for this is as follows:

```python

# This is a dictionary mapping the service names to their respective environment variables.
environment_map = 
{
  # DeepL translation service
  "deepl": "DEEPL_API_KEY",
  
  # Gemini translation service
  "gemini": "GEMINI_API_KEY",
  
  # OpenAI translation service
  "openai": "OPENAI_API_KEY",
  
  # Google Translate service
  "google translate": "PATH_TO_GOOGLE_CREDENTIALS_JSON",
  
  # Anthropic translation service
  "anthropic": "ANTHROPIC_API_KEY",
}

```

---------------------------------------------------------------------------------------------------------------------------------------------------

## **License**<a name="license"></a>

This project, EasyTL, is licensed under the GNU Lesser General Public License v2.1 (LGPLv2.1) - see the LICENSE file for complete details.

The LGPL is a permissive copyleft license that enables this software to be freely used, modified, and distributed. It is particularly designed for libraries, allowing them to be included in both open source and proprietary software. When using or modifying EasyTL, you can choose to release your work under the LGPLv2.1 to contribute back to the community or incorporate it into proprietary software as per the license's permissions.

---------------------------------------------------------------------------------------------------------------------------------------------------

## **Contact**<a name="contact"></a>

If you have any questions or suggestions, feel free to reach out to me at [Bikatr7@proton.me](mailto:Bikatr7@proton.me)

Also feel free to check out the [GitHub repository](https://github.com/Bikatr7/EasyTL) for this project.

Or the issue tracker [here](https://github.com/Bikatr7/EasyTL/issues).

---------------------------------------------------------------------------------------------------------------------------------------------------

## **Contribution**<a name="contribution"></a>

Contributions are welcome! I don't have a specific format for contributions, but please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

---------------------------------------------------------------------------------------------------------------------------------------------------

## **Acknowledgements**<a name="acknowledgements"></a>

EasyTL was originally developed as a part of [Kudasai](https://github.com/Bikatr7/Kudasai), a Japanese preprocessor later turned Machine Translator. It was later split off into its own package to be used independently of Kudasai for multiple reasons.

This package is also my second serious attempt at creating a Python package, so I'm sure there are some things that could be improved. Feedback is welcomed.

---------------------------------------------------------------------------------------------------------------------------------------------------
