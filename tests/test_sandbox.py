import asyncio
import re

import pytest
from fastapi.testclient import TestClient

from agent_platform import tools as registry
from agent_platform.tools import ToolRefused, call, granted, passport_problems
from sandbox import tools
from sandbox.app import app
from sandbox.mcp_server import build_server
from sandbox.store import STORE


@pytest.mark.parametrize(
    ("payer", "customer"),
    [
        ("OKAFOR LOGISTICS", "C-101"),
        ("Tanaka Foods K.K.", "C-103"),
        ("moreau et fils", "C-104"),
        ("J. Smith", None),
        ("", None),
    ],
)
def test_find_customer(payer, customer):
    found = tools.find_customer(payer)
    assert (found["id"] if found else None) == customer


def test_open_invoices_leave_out_paid_ones():
    ids = [i["id"] for i in tools.list_open_invoices("C-102")]
    assert ids == ["INV-1203"]


def test_match_statement_line_checks_the_sums():
    assert "error" in tools.match_statement_line("BSL-5002", ["INV-1201", "INV-1202"])
    ok = tools.match_statement_line("BSL-5002", ["INV-1201", "INV-1202"], fee=12.5)
    assert ok["ok"]
    assert STORE.invoices["INV-1201"].status == "paid"
    assert "already matched" in tools.match_statement_line("BSL-5002", ["INV-1203"])["error"]
    assert len(tools.list_unmatched_statement_lines()) == 7


def test_refunds_never_exceed_what_is_left():
    assert "more than is left" in tools.issue_refund("RR-9001", 60)["error"]
    assert tools.issue_refund("RR-9001", 48)["ok"]
    assert STORE.orders["ORD-7001"].refunded == 48
    assert "no pending" in tools.issue_refund("RR-9001", 1)["error"]
    assert tools.decline_refund("RR-9004", "outside_window")["ok"]
    assert len(tools.list_pending_refund_requests()) == 4


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), -1, 0])
def test_invalid_refund_amounts_leave_balances_unchanged(amount):
    assert "error" in tools.issue_refund("RR-9001", amount)
    assert STORE.orders["ORD-7001"].refunded == 0


def test_match_refuses_repeated_invoices_and_invalid_fees():
    assert "error" in tools.match_statement_line("BSL-5002", ["INV-1201", "INV-1201"])
    assert "error" in tools.match_statement_line("BSL-5002", ["INV-1201"], float("nan"))
    assert "error" in tools.match_statement_line("BSL-5002", ["INV-1201"], -10)
    assert STORE.invoices["INV-1201"].status == "open"


def test_reset_restores_the_seed():
    tools.issue_refund("RR-9001", 48)
    STORE.reset()
    assert STORE.orders["ORD-7001"].refunded == 0


def test_http_api():
    with TestClient(app) as client:
        assert len(client.get("/statement-lines/unmatched").json()) == 8
        assert (
            client.get("/customers", params={"name": "Patel Electronics"}).json()["id"] == "C-105"
        )
        assert client.get("/customers", params={"name": "Nobody"}).status_code == 404
        r = client.post("/statement-lines/BSL-5001/match", json={"invoice_ids": ["INV-1203"]})
        assert r.status_code == 200
        r = client.post("/refund-requests/RR-9003/refund", json={"amount": 999})
        assert r.status_code == 409
        assert "Never more than was paid" in client.get("/policy/refunds").text
        assert client.post("/reset").json() == {"ok": True}
        assert len(client.get("/statement-lines/unmatched").json()) == 8


def test_mcp_server_lists_every_sandbox_tool_with_write_hints():
    server = build_server()
    listed = {t.name: t for t in asyncio.run(server.list_tools())}
    assert set(listed) == set(registry.tools_config().tools)
    assert listed["get_order"].annotations.read_only_hint is True
    assert listed["issue_refund"].annotations.destructive_hint is True


def test_mcp_tool_call_reaches_the_sandbox():
    server = build_server()
    result = asyncio.run(server.call_tool("get_order", {"order_id": "ORD-7003"}))
    assert "ORD-7003" in str(result)


# ── the platform's tool policy ───────────────────────────────────────────────────────────


def test_tools_need_a_grant():
    with pytest.raises(ToolRefused, match="only be called while the platform runs"):
        call("get_order", order_id="ORD-7001")


def test_only_granted_tools():
    with granted("demo", ["get_order"], can_write=False):
        assert call("get_order", order_id="ORD-7001")["id"] == "ORD-7001"
        with pytest.raises(ToolRefused, match="not in the passport"):
            call("list_pending_refund_requests")
        with pytest.raises(ToolRefused, match="no tool called"):
            call("delete_everything")


def test_writes_need_permission_to_write():
    with granted("demo", ["issue_refund"], can_write=False), pytest.raises(ToolRefused):
        call("issue_refund", request_id="RR-9001", amount=48)
    assert STORE.orders["ORD-7001"].refunded == 0
    with granted("demo", ["issue_refund"], can_write=True) as grant:
        assert call("issue_refund", request_id="RR-9001", amount=48)["ok"]
    assert [c.tool for c in grant.calls] == ["issue_refund"]


def test_passport_tool_rules():
    assert passport_problems(["get_order"], "advise") == []
    assert "need autonomy above advise" in passport_problems(["issue_refund"], "advise")[0]
    assert "not in config/tools.yaml" in passport_problems(["fax"], "act_alone")[0]


def test_written_policy_matches_policy_code():
    from agent_platform.settings import ROOT

    text = (ROOT / "sandbox" / "refund_policy.md").read_text(encoding="utf-8")
    code = (ROOT / "agents" / "refund-requests" / "policy.py").read_text(encoding="utf-8")
    numbers = dict(re.findall(r"^([A-Z_]+) = ([\d.]+)$", code, flags=re.M))
    assert float(numbers["ACT_ALONE_UP_TO"]) == 50
    assert "Refunds of 50 or less" in text
    assert f"at least {numbers['NOT_DELIVERED_AFTER_DAYS']} days" in text
    assert f"within {numbers['DAMAGED_WITHIN_DAYS']} days" in text
    assert f"within {numbers['CHANGED_MIND_WITHIN_DAYS']} days" in text
