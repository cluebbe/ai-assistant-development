"""Task 6: does the search find the right chunk for ten test questions?

Only the search is tested here, not the answer. A wrong chunk means the model
never sees the right facts, however good the prompt is.
"""

import json
import time
from pathlib import Path

from kb import TOP_K, search

tests = json.loads(Path("golden_retrieval.json").read_text(encoding="utf-8"))
first, top_k = 0, 0

for test in tests:
    hits = search(test["question"])
    headings = [chunk["heading"] for score, chunk in hits]
    if headings[0] == test["expected"]:
        first += 1
        result = "FIRST"
    elif test["expected"] in headings:
        top_k += 1
        result = f"TOP {TOP_K}"
    else:
        result = "MISSED"
    print(f"{result:7} {test['question']}")
    if result != "FIRST":
        print(f"        expected: {test['expected']}")
        print(f"        found:    {hits[0][1]['source']} > {headings[0]}")
    time.sleep(1)  # Stay below the free plan's requests per minute

print(f"\nFound first: {first}/{len(tests)} · found in top {TOP_K}: "
      f"{first + top_k}/{len(tests)}")
