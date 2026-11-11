"""Decide a refund request against the written policy.

1. precheck  rules that need only numbers: already refunded, more than was paid;
2. classify  the model reads the customer's message and names the reason: damaged, charged
             twice, never arrived, changed their mind, or something else;
3. decide    the policy (policy.py) does the rest: dates, photos, charges, amounts.

The model never decides whether to pay. It only reads the message. Refunds within policy of
50 or less are approved; larger ones are proposed for a person's approval; requests outside
policy are declined with the reason; anything else goes to a support specialist.
"""

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_platform.gateway import chat_json

from .policy import REASONS, decide, handover, precheck

SYSTEM = """You read refund requests sent to a payments company that sells card readers, \
terminals and supplies to merchants. Name the customer's reason with exactly one of:

- damaged: the item arrived broken, cracked or crushed
- duplicate_charge: they were charged more than once for the same order
- not_delivered: the order never arrived
- changed_mind: they no longer want an item that arrived fine
- other: anything else, such as a fault after use, a missing part, a wrong item, a question,
  or a message that mixes several reasons

Reply with JSON only: {"reason": "<one of the five>"}"""


class State(TypedDict, total=False):
    request: dict[str, Any]
    order: dict[str, Any]
    reason: str
    output: dict[str, Any]


def check_numbers(state: State) -> State:
    result = precheck(state["request"], state["order"])
    return {"output": result} if result else {}


def classify(state: State) -> State:
    reply = chat_json(json.dumps({"message": state["request"]["message"]}), system=SYSTEM)
    reason = str(reply.get("reason", "")).strip().lower()
    return {"reason": reason if reason in REASONS else "other"}


def apply_policy(state: State) -> State:
    if state["reason"] not in REASONS:
        return {"output": handover("reason unclear")}
    return {"output": decide(state["reason"], state["request"], state["order"])}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("check_numbers", check_numbers)
    graph.add_node("classify", classify)
    graph.add_node("apply_policy", apply_policy)
    graph.add_edge(START, "check_numbers")
    graph.add_conditional_edges(
        "check_numbers", lambda s: END if "output" in s else "classify", ["classify", END]
    )
    graph.add_edge("classify", "apply_policy")
    graph.add_edge("apply_policy", END)
    return graph.compile()
