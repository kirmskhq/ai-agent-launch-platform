"""Durable decisions, staff reviews, approval and measured value."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from agent_platform.cases import is_handover
from agent_platform.settings import paths
from sandbox.store import TENANT


@contextmanager
def database():
    path = paths().state / "decisions.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS decisions (
        id TEXT PRIMARY KEY, tenant TEXT NOT NULL, agent TEXT NOT NULL,
        created_at TEXT NOT NULL, payload TEXT NOT NULL, output TEXT NOT NULL,
        cost REAL NOT NULL, model TEXT NOT NULL, failed INTEGER NOT NULL,
        reviewed INTEGER, reviewer TEXT, action TEXT, state TEXT NOT NULL,
        action_result TEXT, minutes REAL NOT NULL, hour_cost REAL NOT NULL,
        error_cost REAL NOT NULL)""")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def record(agent, payload, output, cost, failed, tenant="default") -> str:
    passport = agent.passport
    decision_id = uuid4().hex
    action = None
    if not is_handover(output):
        if agent.name == "payment-reconciliation" and output.get("invoice_ids"):
            action = {
                "tool": "match_statement_line",
                "arguments": {
                    "line_id": payload["line"]["id"],
                    "invoice_ids": output["invoice_ids"],
                    "fee": output.get("fee", 0),
                },
            }
        if agent.name == "refund-requests":
            choice = output.get("decision")
            if choice in ("approve", "propose"):
                action = {
                    "tool": "issue_refund",
                    "arguments": {
                        "request_id": payload["request"]["id"],
                        "amount": output["amount"],
                    },
                }
            elif choice == "decline":
                action = {
                    "tool": "decline_refund",
                    "arguments": {
                        "request_id": payload["request"]["id"],
                        "reason": output["reason"],
                    },
                }
    state = "answered"
    if action and passport.autonomy != "advise":
        state = (
            "pending"
            if passport.autonomy == "act_after_approval" or output.get("decision") == "propose"
            else "ready"
        )
    else:
        action = None
    with database() as db:
        db.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,NULL,NULL,?,?,NULL,?,?,?)",
            (
                decision_id,
                tenant,
                agent.name,
                datetime.now(UTC).isoformat(),
                json.dumps(payload),
                json.dumps(output),
                cost,
                passport.model,
                int(failed),
                json.dumps(action) if action else None,
                state,
                passport.baseline.minutes_per_case,
                passport.baseline.staff_cost_per_hour,
                passport.error_cost,
            ),
        )
    return decision_id


def decisions(tenant="default", agent=None) -> list[dict]:
    with database() as db:
        rows = db.execute(
            "SELECT * FROM decisions WHERE tenant=? ORDER BY created_at DESC", (tenant,)
        ).fetchall()
    result = []
    for row in rows:
        value = dict(row)
        if agent and value["agent"] != agent:
            continue
        for key in ("payload", "output", "action", "action_result"):
            value[key] = json.loads(value[key]) if value[key] else None
        result.append(value)
    return result


def review(decision_id, correct, reviewer, tenant="default"):
    if not reviewer.strip():
        raise ValueError("reviewer is required")
    with database() as db:
        changed = db.execute(
            "UPDATE decisions SET reviewed=?, reviewer=? WHERE id=? AND tenant=?",
            (int(correct), reviewer, decision_id, tenant),
        ).rowcount
        if not changed:
            raise LookupError("decision not found")


def execute(decision_id, tenant="default", *, approved_by=None, reject=False):
    from agent_platform.runner import require_running
    from agent_platform.tools import call, granted

    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM decisions WHERE id=? AND tenant=?", (decision_id, tenant)
        ).fetchone()
        if row is None:
            raise LookupError("decision not found")
        if row["state"] not in ("ready", "pending"):
            return dict(row)
        if row["state"] == "pending" and not approved_by:
            raise ValueError("approval requires a staff name")
        agent = require_running(row["agent"])
        if agent.passport.autonomy == "advise":
            raise ValueError("agent may only advise")
        if reject:
            db.execute(
                "UPDATE decisions SET state='rejected', reviewer=? WHERE id=?",
                (approved_by, decision_id),
            )
            return {"state": "rejected"}
        action = json.loads(row["action"])
        previous = db.execute(
            "SELECT id FROM decisions WHERE tenant=? AND agent=? AND action=? "
            "AND state IN ('executed', 'unknown') AND id<>?",
            (tenant, row["agent"], row["action"], decision_id),
        ).fetchone()
        if previous:
            db.execute("UPDATE decisions SET state='duplicate' WHERE id=?", (decision_id,))
            return {"state": "duplicate", "previous_decision": previous["id"]}
        # Commit the claim before the external write. A crash leaves an unknown outcome,
        # which must be reconciled by staff before another attempt.
        db.execute(
            "UPDATE decisions SET state='unknown', reviewer=? WHERE id=?",
            (approved_by, decision_id),
        )
    token = TENANT.set(tenant)
    try:
        with granted(agent.name, agent.passport.tools, can_write=True):
            result = call(action["tool"], **action["arguments"])
        state = "failed" if isinstance(result, dict) and result.get("error") else "executed"
    except Exception as error:
        result, state = {"error": str(error)}, "unknown"
    finally:
        TENANT.reset(token)
    with database() as db:
        db.execute(
            "UPDATE decisions SET state=?, action_result=? WHERE id=?",
            (state, json.dumps(result), decision_id),
        )
    return {"state": state, "result": result}


def measured(agent, tenant="default", weeks=4) -> dict:
    if weeks < 1:
        raise ValueError("weeks must be positive")
    rows = decisions(tenant, agent)
    reviewed = [r for r in rows if r["reviewed"] is not None]
    saving = sum(
        r["minutes"] / 60 * r["hour_cost"]
        for r in reviewed
        if r["reviewed"]
        and not is_handover(r["output"])
        and not r["failed"]
        and r["state"] in ("answered", "executed")
    )
    errors = sum(
        r["error_cost"] for r in reviewed if not r["reviewed"] and not is_handover(r["output"])
    )
    cost = sum(r["cost"] for r in rows)
    due = bool(rows) and datetime.now(UTC) - datetime.fromisoformat(
        rows[-1]["created_at"]
    ) >= timedelta(weeks=weeks)
    net = saving - errors - cost
    coverage = len(reviewed) / len(rows) if rows else 0
    recommendation = "collect evidence"
    reason = f"Observe at least {weeks} weeks and review at least 80% of calls."
    if due and coverage >= 0.8:
        recommendation = "scale up" if net > 0 else "shut down" if net < 0 else "redesign"
        reason = f"Reviewed {len(reviewed)} of {len(rows)} calls. Measured net effect is {net:.2f}."
    return {
        "calls": len(rows),
        "reviewed": len(reviewed),
        "review_coverage": coverage,
        "staff_saving": saving,
        "error_cost": errors,
        "model_cost": cost,
        "net": net,
        "weeks": weeks,
        "checkpoint_due": due,
        "recommendation": recommendation,
        "reason": reason,
    }
