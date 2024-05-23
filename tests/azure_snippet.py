## Copyright Alejandro (https://github.com/alemalvarez)
## Use of this source code is governed by a GNU Lesser General Public License v2.1
## license that can be found in the LICENSE file.

from dotenv import load_dotenv
import os
import asyncio
load_dotenv()

import requests, uuid, json

from easytl.azure_service import AzureService

# Add your key and endpoint
key = os.getenv('AZURE_TRANSLATOR_KEY')
endpoint = os.getenv('AZURE_TRANSLATOR_ENDPOINT')

# location, also known as region.
# required if you're using a multi-service or regional (not global) resource. It can be found in the Azure portal on the Keys and Endpoint page.
location = os.getenv('AZURE_TRANSLATOR_LOCATION')

path = '/translate'
constructed_url = endpoint + path

params = {
    'api-version': '3.0',
    'from': 'en',
    'to': ['fr', 'zu']
}

headers = {
    'Ocp-Apim-Subscription-Key': key,
    # location required if you're using a multi-service or regional (not global) resource.
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

# You can pass more than one object in body.
body = [{
    'text': 'I would really like to drive your car around the block a few times!'
}]

request = requests.post(constructed_url, params=params, headers=headers, json=body)
response = request.json()

print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))

azure = AzureService()

azure._set_credentials(api_key=key, location=location, endpoint=endpoint)
azure._set_attributes(target_language='fr',
                      source_language='en')

async def main():
    translation = await azure._translate_text_async('I would really like to drive your car around the block a few times!')
    print(translation)

asyncio.run(main())