"""Rules first: large amounts always go to a person. The model only names the category."""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_platform.gateway import chat_json

CATEGORIES = {"travel", "equipment", "meals"}
LIMIT = 1000


class State(TypedDict, total=False):
    description: str
    amount: float
    output: dict[str, Any]


def code(state: State) -> State:
    if state["amount"] > LIMIT:
        return {"output": {"handover": True, "reason": f"amount above {LIMIT}"}}
    reply = chat_json(
        f"Expense line: {state['description']}\n"
        'Reply as JSON: {"category": one of travel, equipment, meals, or unclear}'
    )
    category = str(reply.get("category", "")).lower()
    if category not in CATEGORIES:
        return {"output": {"handover": True, "reason": "category unclear"}}
    return {"output": {"category": category}}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("code", code)
    graph.add_edge(START, "code")
    graph.add_edge("code", END)
    return graph.compile()
