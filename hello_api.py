"""Task 3: one request to the API, one answer."""

import sys

from google.genai import types

from config import CHAT_MODEL, get_client

client = get_client()

response = client.models.generate_content(
    model=CHAT_MODEL,
    contents="What does a bicycle chain do? Answer in two sentences.",
    config=types.GenerateContentConfig(
        system_instruction="You are a friendly bike mechanic. Use plain English.",
        temperature=0.7,
    ),
)

print(response.text)

usage = response.usage_metadata
print("---")
print(f"Input tokens:    {usage.prompt_token_count}")
print(f"Thinking tokens: {usage.thoughts_token_count or 0}")
print(f"Output tokens:   {usage.candidates_token_count}")
print(f"Total (billed):  {usage.total_token_count}")

# Run "python hello_api.py --raw" to see the whole response the API sent back
if "--raw" in sys.argv:
    print(response.model_dump_json(indent=2, exclude_none=True))
