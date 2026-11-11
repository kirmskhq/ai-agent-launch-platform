"""Match an incoming bank payment to the open invoices it pays.

Rules first, the model only where rules cannot read the reference, and rules again to check
the model:

1. rules    a reference in the standard form (INV-1234) naming one open invoice of exactly
            the paid amount is matched without the model;
2. model    anything else goes to the model, which reads messy references
            ("inv 1043 + 1044 minus fee") and proposes which invoices are paid;
3. verify   the proposal is accepted only if the invoices exist, are named in the reference
            (or the amount points to exactly one invoice when no number is given) and add up
            to the payment less a plausible bank fee. The fee is computed, not taken from the
            model.

Anything that does not check out goes to finance staff with the reason.
"""

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_platform.gateway import chat_json

MAX_FEE = 25.00
MAX_FEE_SHARE = 0.03
CANONICAL = re.compile(r"\bINV-(\d{4})\b")
NUMBER = re.compile(r"(?<!\d)(\d{4})(?!\d)")

SYSTEM = """You help the finance team of a payments company match incoming bank payments to \
open invoices. You get one bank statement line and the payer's open invoices.

Decide which open invoices this payment settles in full. Payers write references in many \
ways: "inv 1043", "invoice 1043", "1043/1044", "invoices 1043 and 1044 less fee". A bank \
fee of up to 25 may be deducted from the amount.

Rules:
- Pick only invoices from the list.
- If the reference names invoice numbers, pick only invoices it names as being paid now; \
ignore numbers it says were paid before or are not part of this payment.
- If the reference names no invoice number, pick an invoice only if exactly one open \
invoice has exactly the paid amount.
- If this is a part payment, an overpayment, or you are not sure, pick nothing.

Reply with JSON only: {"invoice_ids": ["INV-1043", ...], "note": "one short sentence"}. \
Use an empty list when you pick nothing."""


class State(TypedDict, total=False):
    line: dict[str, Any]
    open_invoices: list[dict[str, Any]]
    proposal: list[str]
    note: str
    output: dict[str, Any]


def _handover(reason: str) -> dict[str, Any]:
    return {"handover": True, "reason": reason}


def _numbers(reference: str) -> set[str]:
    return {f"INV-{n}" for n in NUMBER.findall(reference)}


def rules(state: State) -> State:
    line, invoices = state["line"], state["open_invoices"]
    if not invoices:
        return {"output": _handover("no open invoices for this payer")}
    by_id = {i["id"]: i for i in invoices}
    named = [f"INV-{n}" for n in CANONICAL.findall(line["reference"])]
    if (
        len(named) == 1
        and named[0] in by_id
        and abs(by_id[named[0]]["amount"] - line["amount"]) < 0.005
    ):
        return {"output": {"invoice_ids": named, "fee": 0.0, "method": "rules"}}
    return {}


def ask_model(state: State) -> State:
    reply = chat_json(
        json.dumps({"statement_line": state["line"], "open_invoices": state["open_invoices"]}),
        system=SYSTEM,
    )
    ids = reply.get("invoice_ids") or []
    if not isinstance(ids, list):
        ids = []
    return {"proposal": [str(i) for i in ids], "note": str(reply.get("note", ""))}


def verify(state: State) -> State:
    line, invoices = state["line"], state["open_invoices"]
    by_id = {i["id"]: i for i in invoices}
    ids = sorted(set(state.get("proposal") or []))
    if not ids:
        return {"output": _handover(f"no clear match: {state.get('note') or 'model unsure'}")}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        return {"output": _handover(f"proposed invoices are not open: {', '.join(unknown)}")}

    named = _numbers(line["reference"])
    if named:
        unnamed = [i for i in ids if i not in named]
        if unnamed:
            return {"output": _handover(f"reference does not name {', '.join(unnamed)}")}
    else:
        exact = [i for i in invoices if abs(i["amount"] - line["amount"]) < 0.005]
        if len(exact) != 1 or ids != [exact[0]["id"]]:
            return {"output": _handover("no invoice named and the amount fits no single one")}

    total = round(sum(by_id[i]["amount"] for i in ids), 2)
    fee = round(total - line["amount"], 2)
    if fee < 0:
        return {"output": _handover(f"paid {line['amount']:.2f}, more than the {total:.2f} due")}
    if fee > min(MAX_FEE, MAX_FEE_SHARE * total):
        return {"output": _handover(f"short by {fee:.2f}: a part payment, not a bank fee")}
    return {"output": {"invoice_ids": ids, "fee": fee, "method": "model"}}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("rules", rules)
    graph.add_node("ask_model", ask_model)
    graph.add_node("verify", verify)
    graph.add_edge(START, "rules")
    graph.add_conditional_edges(
        "rules", lambda s: END if "output" in s else "ask_model", ["ask_model", END]
    )
    graph.add_edge("ask_model", "verify")
    graph.add_edge("verify", END)
    return graph.compile()
