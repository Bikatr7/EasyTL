## Copyright Bikatr7 (https://github.com/Bikatr7)
## Use of this source code is governed by an GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

## third-party libraries
from google.cloud import translate_v2 as translate
from google.cloud.translate_v2 import Client

from google.oauth2 import service_account
from google.oauth2.service_account import Credentials

class GoogleTLService:

    
    
    _translator:Client
    _credentials:Credentials

    ## ISO 639-1 language codes
    _target_lang:str = 'en'
    _source_lang:str | None = None
    
    _format:str = 'text'



















def main():
    credentials = service_account.Credentials.from_service_account_file(
        'C:\\Users\\Tetra\\Desktop\\LN\\API keys\\gen-lang-client-0189553349-6c305b2f0118.json')


    translate_client = translate.Client(credentials=credentials)

    text = 'Hello, world!'

    result = translate_client.translate(text, target_language='ru')

    print(result['translatedText'])

if __name__ == '__main__':
    main()