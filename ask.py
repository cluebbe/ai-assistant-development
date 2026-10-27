"""Task 5: answer one question with retrieval-augmented generation (RAG).

    python ask.py "Can I borrow a bike?"
"""

import sys

from google.genai import types

from config import CHAT_MODEL, get_client, load_system_prompt
from kb import build_rag_prompt, search

question = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or input("Question: ")

# 1. Retrieval: find the chunks that match the question best
hits = search(question)

# 2. Augmentation: put them into the message, together with the rules
prompt = build_rag_prompt(question, hits)

# 3. Generation: let the model write the answer
response = get_client().models.generate_content(
    model=CHAT_MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=load_system_prompt(), temperature=0.2
    ),
)

print(response.text)
print("\nSources:")
for n, (score, chunk) in enumerate(hits, start=1):
    print(f"  [{n}] {chunk['source']} > {chunk['heading']}  (similarity {score:.3f})")

if "--show-prompt" in sys.argv:
    print("\n--- What the model received ---\n" + prompt)
