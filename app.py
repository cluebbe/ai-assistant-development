"""Task 10: the assistant as a web app, ready for Hugging Face Spaces.

    python app.py    # Then open http://127.0.0.1:7860
"""

import json
import os
import time
from datetime import date, datetime
from pathlib import Path

import gradio as gr

from assistant import answer

MAX_MESSAGE_LENGTH = 500   # Long messages cost tokens and are often attacks
MAX_TURNS_PER_CHAT = 20    # A real customer rarely needs more
DAILY_LIMIT = 200          # Protects your free API quota from strangers
LOG_FILE = Path("logs/conversations.jsonl")

usage_today = {"day": date.today(), "count": 0}


def log(entry):
    LOG_FILE.parent.mkdir(exist_ok=True)
    entry["time"] = datetime.now().isoformat(timespec="seconds")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def text_of(content):
    """Gradio stores a message as text, as one part or as a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))


def respond(message, history):
    if usage_today["day"] != date.today():
        usage_today.update(day=date.today(), count=0)
    if usage_today["count"] >= DAILY_LIMIT:
        return "The assistant has reached its limit for today. Please call us: 01234 567890."
    if len(message) > MAX_MESSAGE_LENGTH:
        return f"Please keep your message under {MAX_MESSAGE_LENGTH} characters."
    if len(history) >= 2 * MAX_TURNS_PER_CHAT:
        return "This chat is getting long. Please start a new one, or call us: 01234 567890."

    usage_today["count"] += 1
    pairs = [("user" if m["role"] == "user" else "model", text_of(m["content"])) for m in history]
    start = time.perf_counter()
    result = answer(message, pairs[-6:])  # The last three exchanges are enough context
    log({"type": "turn", "question": message, "answer": result["text"],
         "sources": result["sources"], "tool_calls": result["tool_calls"],
         "seconds": round(time.perf_counter() - start, 2)})
    return result["text"]


def feedback(data: gr.LikeData):
    log({"type": "feedback", "liked": data.liked, "answer": text_of(data.value)})


with gr.Blocks(title="Kurbelwerk Assistant") as demo:
    chatbot = gr.Chatbot(height=480, placeholder="Ask about repairs, prices or your ticket number.")
    chatbot.like(feedback)
    gr.ChatInterface(
        fn=respond,
        chatbot=chatbot,
        title="Ritzel — the Kurbelwerk assistant",
        description=(
            "You are chatting with an **AI assistant**. It can make mistakes. "
            "Do not enter personal data. Practice project: Kurbelwerk is an invented business."
        ),
        examples=["Can I borrow a bike?", "Is KW-1042 ready?", "Are you open today?"],
    )

if __name__ == "__main__":
    # On Hugging Face, set APP_PASSWORD as a secret to make the app private
    password = os.environ.get("APP_PASSWORD")
    demo.launch(auth=("kurbelwerk", password) if password else None)
