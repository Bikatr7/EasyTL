import anthropic

with open ("./tests/anthropic.txt", "r", encoding="utf-8") as file:
    api_key = file.read().strip()

client = anthropic.Anthropic(
    api_key=api_key
)

message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    temperature=0.0,
    system="You are a chatbot.",
    messages=[
        {"role": "user", "content": "Respond to this with 1"},
    ]
)

print(message.content[0].text)
