"""Intelligent feature: confidence-gated triage with input validation and a human-review fallback.

Instead of always guessing an intent, the system:
  1. validates the input (empty, non-text, too long, no letters),
  2. predicts an intent with TF-IDF char n-grams + logistic regression,
  3. returns "needs_human_review" when the model is not confident enough.
It never raises on bad input; it returns a result dict with a status.
"""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

LABELS = ["order_status", "cancel_order", "payment_issue", "product_query", "complaint", "other"]
MAX_CHARS = 500
DEFAULT_THRESHOLD = 0.50  # replaced by the value chosen in evaluate.py (see README)


def build_model():
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10),
    )


def validate(text):
    """Return (clean_text, None) or (None, reason)."""
    if not isinstance(text, str):
        return None, "input must be a string"
    t = text.strip()
    if not t:
        return None, "empty message"
    if len(t) > MAX_CHARS:
        return None, f"message too long (>{MAX_CHARS} characters)"
    if not re.search(r"[A-Za-z஀-௿]", t):
        return None, "no letters found (only digits/symbols/emoji)"
    return t, None


class Triage:
    def __init__(self, model, threshold=DEFAULT_THRESHOLD):
        self.model, self.threshold = model, threshold

    @classmethod
    def from_csv(cls, path, threshold=DEFAULT_THRESHOLD):
        df = pd.read_csv(path)
        return cls(build_model().fit(df["text"], df["label"]), threshold)

    def predict(self, text):
        clean, err = validate(text)
        if err:
            return {"status": "invalid_input", "label": None, "confidence": None, "reason": err}
        try:
            probs = self.model.predict_proba([clean])[0]
        except Exception as e:  # never crash the caller
            return {"status": "error", "label": None, "confidence": None, "reason": f"model error: {e}"}
        order = probs.argsort()[::-1]
        top = [(self.model.classes_[i], float(probs[i])) for i in order[:2]]
        if top[0][1] < self.threshold:
            return {"status": "needs_human_review", "label": None, "confidence": top[0][1],
                    "top2": top, "reason": f"confidence below {self.threshold:.2f}"}
        return {"status": "ok", "label": top[0][0], "confidence": top[0][1], "top2": top, "reason": ""}
