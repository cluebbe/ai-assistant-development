"""Optional Task 11: what do the logs say about your assistant?

    python log_report.py
"""

import json
from collections import Counter
from pathlib import Path

LOG_FILE = Path("logs/conversations.jsonl")
DONT_KNOW = ["don't know", "do not know", "not sure", "weiß ich nicht", "01234 567890"]

entries = [json.loads(line) for line in LOG_FILE.read_text(encoding="utf-8").splitlines() if line]
turns = [e for e in entries if e["type"] == "turn"]
votes = [e for e in entries if e["type"] == "feedback"]

print(f"Questions answered: {len(turns)}")
if turns:
    seconds = sorted(t["seconds"] for t in turns)
    print(f"Answer time: median {seconds[len(seconds) // 2]} s, slowest {seconds[-1]} s")
    tools = Counter(c.split("(")[0] for t in turns for c in t["tool_calls"] if "(" in c)
    print(f"Tool calls: {dict(tools) or 'none'}")

if votes:
    liked = sum(v["liked"] is True for v in votes)
    print(f"Feedback: {liked} thumbs up, {len(votes) - liked} thumbs down")
    for v in votes:
        if v["liked"] is not True:
            print(f"  👎 {v['answer'][:120]}")

gaps = [t["question"] for t in turns if any(w in t["answer"].lower() for w in DONT_KNOW)]
print(f"\nPossible gaps in the knowledge base ({len(gaps)}):")
for question in gaps:
    print(f"  - {question}")
