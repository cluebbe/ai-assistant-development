"""Check that the API key works and list the models you can use."""

from google import genai
from google.genai import errors, types

from config import CHAT_MODEL, EMBED_MODEL, get_client

get_client()  # Stops with a clear message if the key is missing
# Few, short retries and a time limit: if something is wrong, we want to see it soon
client = genai.Client(http_options=types.HttpOptions(
    timeout=30_000, retry_options=types.HttpRetryOptions(attempts=3, initial_delay=2)))

print("Models for text generation and embeddings:")
for model in client.models.list():
    actions = model.supported_actions or []
    if "generateContent" in actions or "embedContent" in actions:
        name = model.name.removeprefix("models/")
        print(f"  {name:40} context window: {model.input_token_limit} tokens")

ok = True
print(f"\nTest call with {CHAT_MODEL} (can take a few seconds) ...")
try:
    response = client.models.generate_content(model=CHAT_MODEL, contents="Say hello in five words.")
    print(f"  Answer: {response.text.strip()}")
except Exception as e:
    ok = False
    print(f"  FAILED: {e.message if isinstance(e, errors.APIError) else e}")
    print("  Pick another 'flash' model from the list and change CHAT_MODEL in config.py.")

print(f"Test call with {EMBED_MODEL} ...")
try:
    result = client.models.embed_content(model=EMBED_MODEL, contents="Hello")
    print(f"  Vector with {len(result.embeddings[0].values)} numbers")
except Exception as e:
    ok = False
    print(f"  FAILED: {e.message if isinstance(e, errors.APIError) else e}")
    print("  Pick another 'embedding' model from the list and change EMBED_MODEL in config.py.")

print("\nEverything works." if ok else "\nFix the line(s) above in config.py and run this script again.")
