"""Task 3: a chat in the terminal, with memory and streaming."""

from google.genai import types

from config import CHAT_MODEL, get_client, load_system_prompt

client = get_client()
config = types.GenerateContentConfig(system_instruction=load_system_prompt())

history = []  # The model remembers nothing. We send the whole chat every time.

print("Chat with the assistant. Type 'quit' to stop.")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in {"", "quit", "exit"}:
        break

    history.append(types.Content(role="user", parts=[types.Part(text=question)]))

    print("Assistant: ", end="", flush=True)
    answer = ""
    for chunk in client.models.generate_content_stream(
        model=CHAT_MODEL, contents=history, config=config
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)
            answer += chunk.text
    print()

    history.append(types.Content(role="model", parts=[types.Part(text=answer)]))
