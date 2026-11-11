"""Keyword stand-in for the model, so the fixture runs without a key."""

import json

KEYWORDS = {
    "travel": ("taxi", "train", "flight", "hotel"),
    "equipment": ("laptop", "monitor", "keyboard"),
    "meals": ("lunch", "dinner", "coffee"),
}


def respond(messages: list[dict[str, str]]) -> str:
    text = messages[-1]["content"].lower()
    for category, words in KEYWORDS.items():
        if any(w in text for w in words):
            return json.dumps({"category": category})
    return json.dumps({"category": "unclear"})
