"""Task 2: tokens, context window and temperature.

    python playground.py tokens "some text"   # Count the tokens of a text
    python playground.py tokens --file knowledge/faq.md
    python playground.py window               # Show the model's limits
    python playground.py temperature 0        # Three answers at temperature 0
    python playground.py system "question"    # Ask with system_prompt.md only
"""

import sys
from pathlib import Path

from google.genai import types

from config import CHAT_MODEL, get_client, load_system_prompt

client = get_client()
command = sys.argv[1] if len(sys.argv) > 1 else ""

if command == "tokens" and len(sys.argv) > 2:
    if sys.argv[2] == "--file":
        text = Path(sys.argv[3]).read_text(encoding="utf-8")
    else:
        text = " ".join(sys.argv[2:])
    # Counting tokens is free and does not use the model
    result = client.models.count_tokens(model=CHAT_MODEL, contents=text)
    print(f"{len(text)} characters = {result.total_tokens} tokens")

elif command == "window":
    model = client.models.get(model=CHAT_MODEL)
    print(f"Model:               {model.name.removeprefix('models/')}")
    print(f"Context window:      {model.input_token_limit} tokens (input)")
    print(f"Longest answer:      {model.output_token_limit} tokens (output)")
    print(f"Highest temperature: {model.max_temperature}")

elif command == "temperature" and len(sys.argv) > 2:
    temperature = float(sys.argv[2])
    for attempt in range(1, 4):
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents="Invent a name for a bicycle repair workshop. Reply with the name only.",
            config=types.GenerateContentConfig(temperature=temperature),
        )
        print(f"{attempt}. {response.text.strip()}")

elif command == "system" and len(sys.argv) > 2:
    # The rules from your system prompt, but no knowledge files
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=" ".join(sys.argv[2:]),
        config=types.GenerateContentConfig(system_instruction=load_system_prompt()),
    )
    print(response.text)

else:
    print(__doc__)
