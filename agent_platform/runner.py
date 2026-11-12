"""Running accepted agents: one call on request, or a scheduled job over a batch of inputs.

Only an agent with status `running` runs, which means it passed its exam on the model its
passport names, and nothing in it changed since. That is the whole deployment gate.

In production a failure never reaches the customer as an answer. If the agent's code raises or
the model cannot be reached, the case goes to a person, with the reason attached.
"""

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_platform.cases import is_handover
from agent_platform.gateway import GatewayError, Ledger, track, use_model
from agent_platform.loader import load_graph, load_job
from agent_platform.registry import AgentRecord, find
from agent_platform.settings import paths
from agent_platform.tools import granted
from sandbox.store import TENANT


class NotRunnable(RuntimeError):
    pass


@dataclass(frozen=True)
class Result:
    output: dict
    fell_back: bool
    cost: float
    model_calls: int
    latency_s: float
    decision_id: str = ""
    action_state: str | None = None
    action_result: Any = None


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def require_running(name: str) -> AgentRecord:
    agent = find(name)
    if agent is None:
        raise LookupError(f"no agent called {name!r}")
    if not agent.runs:
        raise NotRunnable(f"{name} does not run: {'; '.join(agent.reasons) or agent.status_label}")
    return agent


def _call(graph: Any, model: str, payload: dict) -> tuple[dict, bool, Ledger, float]:
    started = time.perf_counter()
    with use_model(model), track() as ledger:
        try:
            state = graph.invoke({key: value for key, value in payload.items() if key != "output"})
            output = state.get("output") if isinstance(state, dict) else None
            if not isinstance(output, dict):
                raise TypeError("the agent returned no output object")
            fell_back = False
        except GatewayError as e:
            output, fell_back = {"handover": True, "reason": f"model unavailable: {e}"}, True
        except Exception as e:
            output = {"handover": True, "reason": f"agent failed: {type(e).__name__}: {e}"}
            fell_back = True
    return output, fell_back, ledger, time.perf_counter() - started


def invoke(name: str, payload: dict, *, tenant: str = "default") -> Result:
    token = TENANT.set(tenant)
    try:
        return _invoke(name, payload, tenant=tenant)
    finally:
        TENANT.reset(token)


def _invoke(name: str, payload: dict, *, tenant: str) -> Result:
    agent = require_running(name)
    assert agent.passport is not None
    from agent_platform.production import execute, record

    decision_id = record(
        agent,
        payload,
        {"handover": True, "reason": "in progress"},
        0,
        False,
        tenant,
        in_progress=True,
    )
    started = time.perf_counter()
    try:
        graph = load_graph(agent.path)
        with granted(name, agent.passport.tools, can_write=False):
            output, fell_back, ledger, latency = _call(graph, agent.passport.model, payload)
    except Exception as error:
        output = {"handover": True, "reason": f"agent could not start: {type(error).__name__}"}
        fell_back, ledger, latency = True, Ledger(), time.perf_counter() - started
    record(agent, payload, output, ledger.cost, fell_back, tenant, decision_id=decision_id)
    from agent_platform.production import decisions

    decision = next(d for d in decisions(tenant, name) if d["id"] == decision_id)
    if decision["state"] == "ready":
        execute(decision_id, tenant)
        decision = next(d for d in decisions(tenant, name) if d["id"] == decision_id)
    action_state = decision["state"] if decision["action"] else None
    fell_back = fell_back or action_state in ("failed", "unknown")
    return Result(
        output=output,
        fell_back=fell_back,
        cost=ledger.cost,
        model_calls=len(ledger.calls),
        latency_s=latency,
        decision_id=decision_id,
        action_state=action_state,
        action_result=decision["action_result"],
    )


def _runs_file(name: str) -> Path:
    return paths().state / "runs" / f"{name}.jsonl"


def save_run(name: str, entry: dict) -> None:
    path = _runs_file(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def recent_runs(name: str, n: int = 10) -> list[dict]:
    path = _runs_file(name)
    if not path.is_file():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in reversed(lines[-n:])]


def run_job(name: str, *, trigger: str = "manual", tenant: str = "default") -> dict:
    """job.collect() -> the agent on every input -> a summary in the run history."""
    agent = require_running(name)
    assert agent.passport is not None
    job = load_job(agent.path)
    if job is None:
        raise NotRunnable(f"{name} has no job.py, so there is nothing to run on a schedule")
    entry: dict[str, Any] = {"started_at": now_utc(), "trigger": trigger}
    started = time.perf_counter()
    answered = handed_over = fell_back = 0
    action_counts: dict[str, int] = {}
    cost = 0.0
    token = TENANT.set(tenant)
    try:
        with granted(name, agent.passport.tools, can_write=False):
            payloads = list(job.collect())
    finally:
        TENANT.reset(token)
    for payload in payloads:
        result = invoke(name, payload, tenant=tenant)
        output, failed = result.output, result.fell_back
        cost += result.cost
        fell_back += failed
        if result.action_state:
            state = result.action_state
            action_counts[state] = action_counts.get(state, 0) + 1
        if is_handover(output) or failed:
            handed_over += 1
        else:
            answered += 1
    entry.update(
        finished_at=now_utc(),
        duration_s=round(time.perf_counter() - started, 1),
        cases=answered + handed_over,
        answered=answered,
        handed_over=handed_over,
        fell_back=fell_back,
        action_counts=action_counts,
        cost=round(cost, 6),
        model=agent.passport.model,
    )
    save_run(name, entry)
    return entry
