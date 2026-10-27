"""Task 7: the complete assistant. Instructions + knowledge base + tools.

    python assistant.py    # Chat in the terminal and watch the tool calls
"""

from google.genai import types

from config import CHAT_MODEL, get_client, load_system_prompt
from kb import build_rag_prompt, search
from repair_tools import TOOLS

CONFIG = types.GenerateContentConfig(
    system_instruction=load_system_prompt(),
    tools=TOOLS,       # The SDK describes the functions to the model and runs them
    temperature=0.2,
)


def answer(question, history=()):
    """Answer one question. history: earlier (role, text) pairs, role 'user' or 'model'."""
    hits = search(question)
    contents = [types.Content(role=role, parts=[types.Part(text=text)]) for role, text in history]
    contents.append(types.Content(role="user", parts=[types.Part(text=build_rag_prompt(question, hits))]))

    response = get_client().models.generate_content(model=CHAT_MODEL, contents=contents, config=CONFIG)

    tool_calls = []
    for content in response.automatic_function_calling_history or []:
        for part in content.parts or []:
            if part.function_call:
                tool_calls.append(f"{part.function_call.name}({dict(part.function_call.args or {})})")
            if part.function_response:
                tool_calls.append(f"  -> {part.function_response.response}")

    return {
        "text": response.text or "Sorry, I have no answer. Please call us: 01234 567890.",
        "sources": [f"{c['source']} > {c['heading']}" for score, c in hits],
        "tool_calls": tool_calls,
    }


if __name__ == "__main__":
    history = []
    print("Chat with Ritzel. Type 'quit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"", "quit", "exit"}:
            break
        result = answer(question, history)
        for line in result["tool_calls"]:
            print(f"  [tool] {line}")
        print(f"Ritzel: {result['text']}")
        print(f"  [sources] {' | '.join(result['sources'])}")
        # Keep only the plain question and answer, not the sources, so the chat stays short
        history += [("user", question), ("model", result["text"])]
