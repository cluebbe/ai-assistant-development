"""Task 9: run the golden set and compare with the previous run.

    python evaluate.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

from assistant import answer
from config import PAUSE_SECONDS

RUNS_DIR = Path("eval_runs")


def check(case, text):
    """Pass if at least one must_include term and no must_not_include term is in the answer."""
    lower = text.lower()
    has_required = not case["must_include"] or any(t.lower() in lower for t in case["must_include"])
    has_forbidden = [t for t in case["must_not_include"] if t.lower() in lower]
    return has_required and not has_forbidden


cases = json.loads(Path("golden_set.json").read_text(encoding="utf-8"))
results = {}
for case in cases:
    text = answer(case["question"])["text"]
    passed = check(case, text)
    results[case["id"]] = {"passed": passed, "question": case["question"], "answer": text}
    print(f"{'PASS' if passed else 'FAIL'}  {case['id']:10} {case['question']}")
    if not passed:
        print("      " + text.replace("\n", " ")[:300])
    time.sleep(PAUSE_SECONDS)

# Score per type of test case
print()
for kind in dict.fromkeys(c["type"] for c in cases):
    ids = [c["id"] for c in cases if c["type"] == kind]
    print(f"{kind:10} {sum(results[i]['passed'] for i in ids)}/{len(ids)}")
total = sum(r["passed"] for r in results.values())
print(f"{'total':10} {total}/{len(cases)}")

# Regression check: what changed since the last run?
RUNS_DIR.mkdir(exist_ok=True)
previous_runs = sorted(RUNS_DIR.glob("run-*.json"))
if previous_runs:
    before = json.loads(previous_runs[-1].read_text(encoding="utf-8"))
    for case_id, result in results.items():
        if case_id in before and before[case_id]["passed"] != result["passed"]:
            change = "FIXED     " if result["passed"] else "REGRESSION"
            print(f"{change} {case_id} (compared with {previous_runs[-1].name})")

run_file = RUNS_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}.json"
run_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved to {run_file}")
