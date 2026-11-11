"""A keyword stand-in for the model. It misreads mixed or unusual messages, as keyword
matching does; exams on the mock never count for acceptance."""

import json

KEYWORDS = [
    ("duplicate_charge", ("twice", "two charges", "double", "charged again")),
    ("not_delivered", ("never arrived", "never showed", "not arrived", "still waiting")),
    ("damaged", ("cracked", "broken", "crushed", "smashed", "damaged")),
    ("changed_mind", ("no longer need", "changed our mind", "changed my mind", "don't need")),
]


def respond(messages: list[dict[str, str]]) -> str:
    text = json.loads(messages[-1]["content"])["message"].lower()
    for reason, words in KEYWORDS:
        if any(w in text for w in words):
            return json.dumps({"reason": reason})
    return json.dumps({"reason": "other"})
