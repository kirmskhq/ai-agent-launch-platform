"""What Harbourline's systems let an agent do. Plain functions returning JSON-ready data.

The same functions are served three ways: called in-process by agent jobs (through the
platform's tool registry, which enforces what each agent may call), over HTTP by
`sandbox.app`, and over MCP by `sandbox.mcp_server`.
"""

import re

from sandbox.store import STORE, as_dict

_SUFFIXES = {"ltd", "ab", "kk", "sarl", "sl", "plc", "inc", "llc", "gmbh", "sa", "bv"}


def _normalise(name: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", " ", name.casefold()).split()
    words = [w for w in words if w not in _SUFFIXES and len(w) > 1]
    return " ".join(words)


def find_customer(name: str) -> dict | None:
    """The customer whose name matches a payer name as written on a bank statement."""
    wanted = _normalise(name)
    if not wanted:
        return None
    for c in STORE.customers.values():
        known = _normalise(c.name)
        if wanted == known or known.startswith(wanted) or wanted.startswith(known):
            return as_dict(c)
    return None


def list_open_invoices(customer_id: str) -> list[dict]:
    """Open invoices of one customer, oldest first."""
    invoices = [
        as_dict(i)
        for i in STORE.invoices.values()
        if i.customer_id == customer_id and i.status == "open"
    ]
    return sorted(invoices, key=lambda i: i["issued_on"])


def list_unmatched_statement_lines() -> list[dict]:
    """Bank statement lines not yet matched to invoices."""
    return [as_dict(s) for s in STORE.lines.values() if not s.matched_to]


def get_order(order_id: str) -> dict | None:
    order = STORE.orders.get(order_id)
    return as_dict(order) if order else None


def list_pending_refund_requests() -> list[dict]:
    return [as_dict(r) for r in STORE.refunds.values() if r.status == "pending"]


def get_refund_policy() -> str:
    from pathlib import Path

    return (Path(__file__).parent / "refund_policy.md").read_text(encoding="utf-8")


def match_statement_line(line_id: str, invoice_ids: list[str], fee: float = 0.0) -> dict:
    """Record that a statement line pays these invoices (less a bank fee) and mark them paid."""
    with STORE.lock:
        line = STORE.lines.get(line_id)
        if line is None:
            return {"error": f"no statement line {line_id}"}
        if line.matched_to:
            return {"error": f"{line_id} is already matched to {', '.join(line.matched_to)}"}
        invoices = [STORE.invoices.get(i) for i in invoice_ids]
        if not invoice_ids or any(i is None or i.status != "open" for i in invoices):
            return {"error": "every invoice must exist and be open"}
        total = round(sum(i.amount for i in invoices) - fee, 2)
        if abs(total - line.amount) > 0.005:
            return {"error": f"invoices less fee come to {total}, the line is {line.amount}"}
        for i in invoices:
            i.status = "paid"
        line.matched_to, line.fee = list(invoice_ids), fee
        return {"ok": True, "line": as_dict(line)}


def issue_refund(request_id: str, amount: float) -> dict:
    """Pay a refund. It settles after 24 hours and can be cancelled until then."""
    with STORE.lock:
        request = STORE.refunds.get(request_id)
        if request is None or request.status != "pending":
            return {"error": f"no pending refund request {request_id}"}
        order = STORE.orders[request.order_id]
        if amount <= 0 or amount > round(order.amount - order.refunded, 2):
            return {"error": f"{amount} is more than is left to refund on {order.id}"}
        order.refunded = round(order.refunded + amount, 2)
        request.status, request.note = "refunded", f"refunded {amount:.2f}"
        return {"ok": True, "request": as_dict(request)}


def decline_refund(request_id: str, reason: str) -> dict:
    with STORE.lock:
        request = STORE.refunds.get(request_id)
        if request is None or request.status != "pending":
            return {"error": f"no pending refund request {request_id}"}
        request.status, request.note = "declined", reason
        return {"ok": True, "request": as_dict(request)}
