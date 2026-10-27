"""Check that the API key works and list the models you can use."""

from config import CHAT_MODEL, EMBED_MODEL, get_client

client = get_client()

print("Models for text generation and embeddings:")
for model in client.models.list():
    actions = model.supported_actions or []
    if "generateContent" in actions or "embedContent" in actions:
        name = model.name.removeprefix("models/")
        print(f"  {name:40} context window: {model.input_token_limit} tokens")

response = client.models.generate_content(model=CHAT_MODEL, contents="Say hello in five words.")
print(f"\nTest call with {CHAT_MODEL}: {response.text.strip()}")

result = client.models.embed_content(model=EMBED_MODEL, contents="Hello")
print(f"Test call with {EMBED_MODEL}: {len(result.embeddings[0].values)} numbers")
print("\nEverything works.")
