"""Harbourline Payments sandbox over HTTP. Start with `uv run agents sandbox`."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from sandbox import tools
from sandbox.store import STORE

app = FastAPI(
    title="Harbourline Payments sandbox",
    description="A fictional payments company. In-memory data; restart or POST /reset to undo.",
)


class MatchRequest(BaseModel):
    invoice_ids: list[str]
    fee: float = 0.0


class RefundRequest(BaseModel):
    amount: float


class DeclineRequest(BaseModel):
    reason: str


def _ok(result: dict) -> dict:
    if "error" in result:
        raise HTTPException(409, result["error"])
    return result


@app.get("/customers")
def customers(name: str | None = None) -> list[dict] | dict:
    if name is not None:
        found = tools.find_customer(name)
        if found is None:
            raise HTTPException(404, f"no customer matches {name!r}")
        return found
    return [c.__dict__ for c in STORE.customers.values()]


@app.get("/customers/{customer_id}/open-invoices")
def open_invoices(customer_id: str) -> list[dict]:
    return tools.list_open_invoices(customer_id)


@app.get("/statement-lines/unmatched")
def unmatched_lines() -> list[dict]:
    return tools.list_unmatched_statement_lines()


@app.post("/statement-lines/{line_id}/match")
def match(line_id: str, body: MatchRequest) -> dict:
    return _ok(tools.match_statement_line(line_id, body.invoice_ids, body.fee))


@app.get("/orders/{order_id}")
def order(order_id: str) -> dict:
    found = tools.get_order(order_id)
    if found is None:
        raise HTTPException(404, f"no order {order_id}")
    return found


@app.get("/refund-requests/pending")
def pending_refunds() -> list[dict]:
    return tools.list_pending_refund_requests()


@app.post("/refund-requests/{request_id}/refund")
def refund(request_id: str, body: RefundRequest) -> dict:
    return _ok(tools.issue_refund(request_id, body.amount))


@app.post("/refund-requests/{request_id}/decline")
def decline(request_id: str, body: DeclineRequest) -> dict:
    return _ok(tools.decline_refund(request_id, body.reason))


@app.get("/policy/refunds", response_class=PlainTextResponse)
def refund_policy() -> str:
    return tools.get_refund_policy()


@app.post("/reset")
def reset() -> dict:
    STORE.reset()
    return {"ok": True}
