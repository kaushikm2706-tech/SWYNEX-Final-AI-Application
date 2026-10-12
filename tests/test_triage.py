import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from triage import Triage, validate

tri = Triage.from_csv("data/messages.csv", json.load(open("data/threshold.json"))["threshold"])


def test_bad_inputs_never_raise():
    for bad in ["", "   ", None, 123, "12345", "😀😀", "a" * 600]:
        r = tri.predict(bad)
        assert r["status"] == "invalid_input" and r["label"] is None


def test_clear_message_is_answered():
    r = tri.predict("order cancel pannunga please")
    assert r["status"] == "ok" and r["label"] == "cancel_order"


def test_low_confidence_goes_to_human():
    strict = Triage(tri.model, threshold=0.99)
    assert strict.predict("anna ennoda parcel enga")["status"] == "needs_human_review"


def test_validate():
    assert validate("hi")[1] is None and validate("")[1] == "empty message"
