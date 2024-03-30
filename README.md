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
EasyTL.validate_api_key("deepL")

EasyTL.deepl_translate("私は日本語が話せます", "EN") ## Text to translate, language to translate to, only two "required" arguments but there are more optional arguments for additional functionality.

## easytl also has a generic translate method, which defaults to deepl, requires text, and kwargs for the translate method.

EasyTL.translate("私は日本語が話せます", target_lang="EN")

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

This project (EasyTL) is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE.md) file for details.

The GPL is a copyleft license that promotes the principles of open-source software. It ensures that any derivative works based on this project must also be distributed under the same GPL license. This license grants you the freedom to use, modify, and distribute the software.

Please note that this information is a brief summary of the GPL. For a detailed understanding of your rights and obligations under this license, please refer to the [full license text](LICENSE.md).

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
