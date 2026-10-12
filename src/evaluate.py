"""Evaluation with a held-out test set. The threshold is chosen WITHOUT looking at the test set."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from triage import build_model, Triage, LABELS

df = pd.read_csv("data/messages.csv")
train, test = train_test_split(df, test_size=0.25, stratify=df["label"], random_state=7)
print(f"train={len(train)} test={len(test)}")

# 1) choose threshold using out-of-fold probabilities on TRAIN only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
oof = cross_val_predict(build_model(), train["text"], train["label"], cv=cv, method="predict_proba")
classes = np.array(sorted(train["label"].unique()))
oof_pred, oof_conf = classes[oof.argmax(1)], oof.max(1)
rows = []
for t in np.arange(0.20, 0.81, 0.05):
    m = oof_conf >= t
    rows.append((round(float(t), 2), float(m.mean()), float((oof_pred[m] == train["label"].values[m]).mean()) if m.any() else float("nan")))
print("\nthreshold  coverage  accuracy_on_answered   (out-of-fold, train only)")
for t, c, a in rows:
    print(f"  {t:.2f}      {c:.2f}      {a:.2f}")
# rule: smallest threshold whose accuracy on answered messages is >= 0.85
chosen = next((t for t, c, a in rows if a == a and a >= 0.85), rows[-1][0])
print("\nchosen threshold:", chosen)

# 2) final model on train, evaluate on untouched test
model = build_model().fit(train["text"], train["label"])
tri = Triage(model, chosen)
res = [tri.predict(t) for t in test["text"]]
test = test.assign(pred=[r["label"] for r in res], status=[r["status"] for r in res], conf=[r["confidence"] for r in res])
plain = model.predict(test["text"])
print("\n== Held-out test, ALWAYS-ANSWER baseline ==")
print(classification_report(test["label"], plain, zero_division=0))
ans = test[test.status == "ok"]
print("== Held-out test, WITH confidence gate ==")
print(f"coverage (answered automatically): {len(ans)/len(test):.2f}  ({len(ans)}/{len(test)})")
print(f"accuracy on answered: {(ans.pred == ans.label).mean():.2f}")
print(f"accuracy always-answer: {(plain == test.label).mean():.2f}   macro-F1 always-answer: {f1_score(test.label, plain, average='macro'):.2f}")
print("\nConfusion matrix (always-answer), rows=true, cols=pred, order:", list(model.classes_))
print(confusion_matrix(test.label, plain, labels=model.classes_))
print("\nMisclassified test messages:")
for _, r in test[plain != test.label].iterrows():
    print(f"  '{r.text}'  true={r.label}  pred={plain[test.index.get_loc(_)]}")
json.dump({"threshold": chosen}, open("data/threshold.json", "w"))

# 3) more stable estimate: repeated stratified 5-fold CV on ALL data (always-answer model)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
scores = cross_val_score(build_model(), df["text"], df["label"], scoring="f1_macro",
                         cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7))
print(f"\nRepeated 5-fold CV macro-F1 on all {len(df)} messages: mean={scores.mean():.2f}  std={scores.std():.2f}  min={scores.min():.2f}")
