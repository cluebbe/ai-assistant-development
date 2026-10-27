"""Settings shared by all workshop scripts."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # Reads GEMINI_API_KEY from the file .env, if there is one

# The SDK prints a hint about tool calls that is not relevant for this workshop
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# Model names change over time. If a script says a model is not found or
# "high demand", run check_setup.py and pick another one from the list
# (a "flash" model for chat, an "embedding" model for embeddings).
# Free plans have a daily request limit per model. Flash-Lite models have a
# much higher one than the bigger Flash models, which is why we use one here.
# "-latest" always points to Google's newest version. For a real product, use
# a fixed name, so the behaviour does not change without you noticing.
CHAT_MODEL = "gemini-flash-lite-latest"
EMBED_MODEL = "gemini-embedding-001"

# Free plans allow only a few requests per minute. Scripts that send many
# requests in a row wait this many seconds between them.
PAUSE_SECONDS = 10


_client = None


def get_client():
    """Return the API client, or stop with a clear message if the key is missing."""
    global _client
    if _client is None:
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit(
                "GEMINI_API_KEY is missing. Put it into the file .env "
                "(see 'Before the Workshop')."
            )
        # Picks up GEMINI_API_KEY by itself. If the free plan says "too many
        # requests" (error 429), wait and try again, up to four times.
        _client = genai.Client(http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=4, initial_delay=10)))
    return _client


def load_system_prompt(path="system_prompt.md"):
    """Return the assistant's instructions from system_prompt.md."""
    return Path(path).read_text(encoding="utf-8").strip()
