from types import SimpleNamespace

import pytest

from agent_platform.passport import load_passport
from agent_platform.production import database, decisions, execute, measured, record, review
from sandbox.store import STORE, TENANT


def agent(workspace):
    return SimpleNamespace(
        name="expense-coder",
        path=workspace / "agents" / "expense-coder",
        passport=load_passport(workspace / "agents" / "expense-coder"),
    )


def test_reviewed_saving_and_handover(workspace):
    a = agent(workspace)
    first = record(a, {}, {"category": "travel"}, 0.01, False)
    second = record(a, {}, {"handover": True}, 0.02, False)
    review(first, True, "Dana")
    review(second, True, "Dana")
    value = measured(a.name)
    assert value["staff_saving"] == 2
    assert value["net"] == pytest.approx(1.97)
    assert value["recommendation"] == "collect evidence"


def test_tenant_review_scope(workspace):
    decision = record(agent(workspace), {}, {}, 0, False, "team-a")
    assert decisions("team-b") == []
    with pytest.raises(LookupError):
        review(decision, True, "Dana", "team-b")
    with pytest.raises(LookupError):
        execute(decision, "team-b", approved_by="Dana")


def test_value_checkpoint_uses_observation_window(workspace):
    a = agent(workspace)
    decision = record(a, {}, {}, 0, False)
    review(decision, False, "Dana")
    with database() as db:
        db.execute("UPDATE decisions SET created_at='2025-01-01T00:00:00+00:00'")
    assert measured(a.name)["recommendation"] == "shut down"


def test_sandbox_tenants_have_separate_balances(workspace):
    original = STORE.orders["ORD-2001"].refunded if "ORD-2001" in STORE.orders else None
    token = TENANT.set("separate-company")
    try:
        order = next(iter(STORE.orders.values()))
        order.refunded = 123
    finally:
        TENANT.reset(token)
    assert all(o.refunded != 123 for o in STORE.orders.values())
    if original is not None:
        assert STORE.orders["ORD-2001"].refunded == original


def test_approval_executes_once_and_pending_saves_nothing(workspace, monkeypatch):
    from agent_platform import runner, tools

    a = agent(workspace)
    a.name = "refund-requests"
    a.passport = a.passport.model_copy(
        update={"autonomy": "act_after_approval", "tools": ["issue_refund"]}
    )
    monkeypatch.setattr(runner, "require_running", lambda name: a)
    calls = []
    monkeypatch.setattr(tools, "call", lambda *args, **kwargs: calls.append(kwargs) or {"ok": True})
    output = {"decision": "propose", "amount": 100, "reason": "in_policy"}
    first = record(a, {"request": {"id": "RR-1"}}, output, 0, False)
    review(first, True, "Dana")
    assert measured(a.name)["staff_saving"] == 0
    with pytest.raises(ValueError, match="staff name"):
        execute(first)
    assert execute(first, approved_by="Dana")["state"] == "executed"
    execute(first, approved_by="Dana")
    second = record(a, {"request": {"id": "RR-1"}}, output, 0, False)
    assert execute(second, approved_by="Dana")["state"] == "duplicate"
    assert len(calls) == 1
    assert measured(a.name)["staff_saving"] == 2


def test_changed_agent_cannot_execute_old_approval(workspace, monkeypatch):
    from agent_platform import runner

    a = agent(workspace)
    a.name = "refund-requests"
    a.passport = a.passport.model_copy(update={"autonomy": "act_after_approval"})
    monkeypatch.setattr(runner, "require_running", lambda name: a)
    decision = record(
        a, {"request": {"id": "RR-1"}}, {"decision": "propose", "amount": 100}, 0, False
    )
    a.passport = a.passport.model_copy(update={"error_cost": 8})
    with pytest.raises(ValueError, match="changed since"):
        execute(decision, approved_by="Dana")
