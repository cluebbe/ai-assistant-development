"""Task 9: run the golden set and compare with the previous run.

    python evaluate.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

from google.genai import types
from pydantic import BaseModel

from assistant import answer
from config import CHAT_MODEL, PAUSE_SECONDS, get_client

RUNS_DIR = Path("eval_runs")


class Verdict(BaseModel):
    passed: bool
    reason: str


def check(case, text):
    """Pass if at least one must_include term and no must_not_include term is in the answer."""
    lower = text.lower()
    has_required = not case["must_include"] or any(t.lower() in lower for t in case["must_include"])
    has_forbidden = [t for t in case["must_not_include"] if t.lower() in lower]
    return has_required and not has_forbidden


def judge(case, text):
    """Ask a second model whether the answer meets the case's "judge" criterion."""
    response = get_client().models.generate_content(
        model=CHAT_MODEL,
        contents=(f"Customer question: {case['question']}\n\n<answer>\n{text}\n</answer>\n\n"
                  f"Criterion: {case['judge']}"),
        config=types.GenerateContentConfig(
            system_instruction="You check answers of a customer service assistant strictly. "
                               "Decide whether the answer meets the criterion.",
            response_mime_type="application/json",
            response_schema=Verdict,
            temperature=0,
        ),
    )
    return response.parsed


cases = json.loads(Path("golden_set.json").read_text(encoding="utf-8"))
results = {}
for case in cases:
    text = answer(case["question"])["text"]
    passed = check(case, text)
    reason = "" if passed else "word check failed"
    if passed and "judge" in case:  # Words are fine; now let a model judge the meaning
        time.sleep(PAUSE_SECONDS)
        verdict = judge(case, text)
        passed, reason = verdict.passed, f"judge: {verdict.reason}"
    results[case["id"]] = {"passed": passed, "question": case["question"], "answer": text,
                           "reason": reason}
    print(f"{'PASS' if passed else 'FAIL'}  {case['id']:10} {case['question']}")
    if not passed:
        print(f"      {reason}")
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
