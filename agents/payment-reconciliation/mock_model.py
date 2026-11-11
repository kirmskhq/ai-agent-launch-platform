"""A crude stand-in for the model: picks every invoice whose number appears in the reference.

It does not understand "already paid" or "not this one", so it does worse than a real model on
those cases, which is the point: exams on the mock never count for acceptance.
"""

import json
import re


def respond(messages: list[dict[str, str]]) -> str:
    data = json.loads(messages[-1]["content"])
    line, invoices = data["statement_line"], data["open_invoices"]
    numbers = set(re.findall(r"(?<!\d)(\d{4})(?!\d)", line["reference"]))
    picked = [i["id"] for i in invoices if i["id"].removeprefix("INV-") in numbers]
    if not numbers:
        exact = [i["id"] for i in invoices if abs(i["amount"] - line["amount"]) < 0.005]
        picked = exact if len(exact) == 1 else []
    return json.dumps({"invoice_ids": picked, "note": "numbers found in the reference"})
