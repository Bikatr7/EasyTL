from google.cloud import translate_v2 as translate

from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    'C:\\Users\\Tetra\\Desktop\\LN\\API keys\\gen-lang-client-0189553349-6c305b2f0118.json')


translate_client = translate.Client(credentials=credentials)

text = 'Hello, world!'

result = translate_client.translate(text, target_language='ru')

print(result['translatedText'])