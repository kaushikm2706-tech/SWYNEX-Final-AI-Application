"""Run deliberately difficult inputs through the full triage pipeline and show what happens.
Outcome per case: CORRECT (acceptable label / correct rejection), DEFERRED (sent to human review), WRONG."""
import json
import pandas as pd
from triage import Triage

thr = json.load(open("data/threshold.json"))["threshold"]
tri = Triage.from_csv("data/messages.csv", thr)
hc = pd.read_csv("data/hard_cases.csv", keep_default_na=False)
counts = {"CORRECT": 0, "DEFERRED": 0, "WRONG": 0}
for _, r in hc.iterrows():
    res = tri.predict(r.text)
    got = res["label"] if res["status"] == "ok" else res["status"]
    if got in r.expected.split("|"):
        outcome = "CORRECT"
    elif res["status"] == "needs_human_review" and r.expected != "invalid_input":
        outcome = "DEFERRED"
    else:
        outcome = "WRONG"
    counts[outcome] += 1
    conf = f"{res['confidence']:.2f}" if res["confidence"] is not None else "-"
    print(f"{outcome:8s} | {r.why:28s} | in={r.text!r:50.50s} | out={got} ({conf}) | expected={r.expected}")
print("\n" + ", ".join(f"{k}: {v}" for k, v in counts.items()) + f"  (of {len(hc)})")
