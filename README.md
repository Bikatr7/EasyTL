---------------------------------------------------------------------------------------------------------------------------------------------------
**Table of Contents**

- [Quick Start](#quick-start)
- [Installation](#installation)
- [License](#license)
- [Contact](#contact)
- [Contribution](#contribution)
- [Notes](#notes)

---------------------------------------------------------------------------------------------------------------------------------------------------

## EasyTL

Wrapper for OpenAI, DeepL, and Gemini APIs for easy translation of text.

---------------------------------------------------------------------------------------------------------------------------------------------------
**Quick Start**<a name="quick-start"></a>

To get started with EasyTL, install the package via pip:

```bash
pip install easytl
```

Then, you can translate Japanese text using DeepL by importing the global client.

```python
from easytl import EasyTL

## set your api key
EasyTL.set_api_key("deepL", "your_api_key_here")

## you can also validate your api keys, translation functions will do this automatically
is_valid, e = EasyTL.validate_api_key("deepL")

translated_text = EasyTL.deepl_translate("私は日本語が話せます", "EN") ## Text to translate, language to translate to, only two "required" arguments but there are more optional arguments for additional functionality.

## easytl also has a generic translate method, which defaults to deepl, requires text, and kwargs for the translate method it uses.

translated_text = EasyTL.translate("私は日本語が話せます", target_lang="EN")

## There is also a calculate cost method.
## for deepl, model is deepl
num_characters, cost, model = EasyTL.calculate_cost("私は日本語が話せます", "deepL") ## Text to translate, service to use

## or for llms
## last 2 arguments are optional, defaults to gpt-4 and "Translate to English"
num_tokens, cost, model = EasyTL.calculate_cost("私は日本語が話せます", "openai", "gpt-4", "Translate to English") ## Text to translate, service to use, model to use, prompt to use

```

---------------------------------------------------------------------------------------------------------------------------------------------------

**Installation**<a name="installation"></a>

Python 3.8+

EasyTL can be installed using pip:


```bash
pip install easytl
```

This will install EasyTL along with its dependencies and requirements.

These are the dependencies/requirements that will be installed:
```
setuptools>=61.0

wheel

setuptools_scm>=6.0

tomli

google-generativeai==0.4.1

deepl==1.16.1

openai==1.13.3

backoff==2.2.1

tiktoken==0.6.0

```
---------------------------------------------------------------------------------------------------------------------------------------------------

**License**<a name="license"></a>

This project, EasyTL, is licensed under the GNU Lesser General Public License v2.1 (LGPLv2.1) - see the LICENSE file for complete details.

The LGPL is a permissive copyleft license that enables this software to be freely used, modified, and distributed. It is particularly designed for libraries, allowing them to be included in both open source and proprietary software. When using or modifying EasyTL, you can choose to release your work under the LGPLv2.1 to contribute back to the community or incorporate it into proprietary software as per the license's permissions.

Under the LGPLv2.1, any modifications made to EasyTL's libraries must be shared under the same license, ensuring the open source nature of the library itself is maintained. However, it allows the wider work that includes the library to remain under different licensing terms, provided the terms of the LGPLv2.1 are met for the library.

I encourage the use of EasyTL within all projects. This approach ensures a balance between open collaboration and the flexibility required for proprietary development.

For a thorough explanation of the conditions and how they may apply to your use of the project, please review the full license text. This comprehensive documentation will offer guidance on your rights and obligations when utilizing LGPLv2.1 licensed software.

---------------------------------------------------------------------------------------------------------------------------------------------------

**Contact**<a name="contact"></a>

If you have any questions or suggestions, feel free to reach out to me at [Tetralon07@gmail.com](mailto:Tetralon07@gmail.com).

Also feel free to check out the [GitHub repository](https://github.com/Bikatr7/EasyTL) for this project.

Or the issue tracker [here](https://github.com/Bikatr7/EasyTL/issues).

---------------------------------------------------------------------------------------------------------------------------------------------------

**Contribution**<a name="contribution"></a>

Contributions are welcome! I don't have a specific format for contributions, but please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

---------------------------------------------------------------------------------------------------------------------------------------------------

**Notes**<a name="notes"></a>

EasyTL was originally developed as a part of [Kudasai](https://github.com/Bikatr7/Kudasai), a Japanese preprocessor later turned Machine Translator. It was later split off into its own package to be used independently of Kudasai for multiple reasons.

This package is also my second serious attempt at creating a Python package, so I'm sure there are some things that could be improved. Feedback is welcomed.

---------------------------------------------------------------------------------------------------------------------------------------------------
