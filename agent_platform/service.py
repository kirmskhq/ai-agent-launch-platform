"""The agent service: catalogue pages, a JSON API, and the endpoints that run agents.

    GET  /                                catalogue
    GET  /agents/{name}                   one agent
    GET  /api/agents                      catalogue as JSON
    GET  /api/agents/{name}               one agent as JSON
    GET  /proposals/{name}                a proposal at the readiness checkpoint
    GET  /api/proposals                   proposals as JSON
    POST /api/agents/{name}/invoke        one call: {"input": {...}}          service token
    POST /api/agents/{name}/run           the agent's scheduled job, now      service token
    GET  /api/decisions                   the caller's company's decisions    any token
    GET  /api/agents/{name}/value         measured value                      any token
    POST /api/decisions/{id}/review       {"correct": true}                   reviewer
    POST /api/decisions/{id}/approval     {"reject": false}                   approver
    GET  /api/me                          who the token belongs to            any token
    GET  /operations                      staff page for reviews and approvals
    GET  /healthz                         liveness

Reading the catalogue is open; everything else needs `Authorization: Bearer <token>`. Tokens and
roles are described in `agent_platform.auth`. With no token configured at all, those endpoints
refuse every request rather than run open.

The scheduler starts with the service when AGENTS_SCHEDULER=1.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from agent_platform.auth import Credentials, Permission, Principal
from agent_platform.catalogue import render_agent, render_index, render_proposal
from agent_platform.production import decisions, execute, measured, review
from agent_platform.readiness import as_dict as readiness_dict
from agent_platform.registry import AgentRecord, find, find_proposal, proposals, records
from agent_platform.runner import NotRunnable, invoke, run_job
from agent_platform.scheduler import Scheduler
from agent_platform.settings import load_dotenv

log = logging.getLogger("agent_platform.service")
_bearer = HTTPBearer(auto_error=False)

REFUSALS: dict[Permission, str] = {
    "run": "only a service token can run agents",
    "read": "this token cannot read decisions",
    "review": "reviews need a personal staff token (AGENTS_STAFF_TOKENS)",
    "approve": "approvals need a personal staff token with the approver role",
}


def caller(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    known: Credentials = request.app.state.credentials
    if not known.configured:
        raise HTTPException(503, "no tokens are configured; authenticated endpoints are disabled")
    principal = known.identify(credentials.credentials) if credentials else None
    if principal is None:
        raise HTTPException(401, "missing or wrong bearer token")
    return principal


def allowed(permission: Permission):
    """A dependency that admits only callers whose role carries `permission`."""

    def dependency(principal: Annotated[Principal, Depends(caller)]) -> Principal:
        if not principal.can(permission):
            raise HTTPException(403, REFUSALS[permission])
        return principal

    return dependency


Caller = Annotated[Principal, Depends(caller)]
Runner = Annotated[Principal, Depends(allowed("run"))]
Reader = Annotated[Principal, Depends(allowed("read"))]
Reviewer = Annotated[Principal, Depends(allowed("review"))]
Approver = Annotated[Principal, Depends(allowed("approve"))]


class ReviewRequest(BaseModel):
    correct: bool


class ApprovalRequest(BaseModel):
    reject: bool = False


class InvokeRequest(BaseModel):
    input: dict[str, Any]


def _summary(a: AgentRecord) -> dict:
    p, r = a.passport, a.report
    return {
        "name": a.name,
        "title": p.title if p else None,
        "department": p.department if p else None,
        "owner": p.owner if p else None,
        "model": p.model if p else None,
        "status": a.status,
        "reasons": a.reasons,
        "autonomy": p.autonomy if p else None,
        "readiness": readiness_dict(a.readiness) if a.readiness else None,
        "exam": None
        if r is None
        else {
            "model": r.model,
            "finished_at": r.finished_at,
            "passed": r.passed,
            "answer_accuracy": r.answer_accuracy,
            "accuracy_lower_bound": r.accuracy_lower_bound,
            "min_accuracy": r.min_accuracy,
            "answered_share": r.answered_share,
            "consistency": r.consistency,
            "counts": r.counts,
            "cost_per_case": r.cost_per_case,
            "projected_monthly_net": r.effect["net"],
            "projected_monthly_net_at_lower_bound": r.effect_at_lower_bound["net"],
        },
    }


def _agent(name: str) -> AgentRecord:
    agent = find(name)
    if agent is None:
        raise HTTPException(404, f"no agent called {name!r}")
    return agent


def create_app(*, scheduler: bool | None = None) -> FastAPI:
    """The service. Raises CredentialsError when the token settings are invalid."""
    load_dotenv()
    credentials = Credentials.from_env()
    start_scheduler = os.environ.get("AGENTS_SCHEDULER") == "1" if scheduler is None else scheduler

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sched = Scheduler()
        if start_scheduler:
            sched.start()
            log.info("scheduler started (UTC)")
        try:
            yield
        finally:
            sched.stop()

    app = FastAPI(title="Agent launch platform", lifespan=lifespan)
    app.state.credentials = credentials

    @app.get("/operations", response_class=HTMLResponse)
    def operations_page():
        return (Path(__file__).parent / "templates" / "operations.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_index(records(), proposals())

    @app.get("/agents/{name}", response_class=HTMLResponse)
    def agent_page(name: str) -> str:
        return render_agent(_agent(name))

    @app.get("/proposals/{name}", response_class=HTMLResponse)
    def proposal_page(name: str) -> str:
        readiness = find_proposal(name)
        if readiness is None:
            raise HTTPException(404, f"no proposal called {name!r}")
        return render_proposal(readiness)

    @app.get("/api/proposals")
    def api_proposals() -> list[dict]:
        return [readiness_dict(r) for r in proposals()]

    @app.get("/api/agents")
    def api_agents() -> list[dict]:
        return [_summary(a) for a in records()]

    @app.get("/api/agents/{name}")
    def api_agent(name: str) -> dict:
        return _summary(_agent(name))

    @app.get("/api/me")
    def api_me(principal: Caller) -> dict:
        return {
            "name": principal.name,
            "tenant": principal.tenant,
            "role": principal.role,
            "may": [p for p in ("run", "read", "review", "approve") if principal.can(p)],
        }

    @app.post("/api/agents/{name}/invoke")
    async def api_invoke(name: str, body: InvokeRequest, principal: Runner) -> dict:
        _agent(name)
        try:
            result = await run_in_threadpool(invoke, name, body.input, tenant=principal.tenant)
        except NotRunnable as e:
            raise HTTPException(409, str(e)) from e
        return {
            "decision_id": result.decision_id,
            "output": result.output,
            "action_state": result.action_state,
            "action_result": result.action_result,
            "fell_back_to_person": result.fell_back,
            "cost": round(result.cost, 6),
            "model_calls": result.model_calls,
            "latency_s": round(result.latency_s, 3),
        }

    @app.post("/api/agents/{name}/run")
    async def api_run(name: str, principal: Runner) -> dict:
        _agent(name)
        try:
            return await run_in_threadpool(run_job, name, trigger="api", tenant=principal.tenant)
        except NotRunnable as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/decisions")
    def api_decisions(principal: Reader) -> list[dict]:
        return decisions(principal.tenant)

    @app.post("/api/decisions/{decision_id}/review")
    def api_review(decision_id: str, body: ReviewRequest, principal: Reviewer) -> dict:
        assert principal.name is not None
        try:
            review(decision_id, body.correct, principal.name, principal.tenant)
        except LookupError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"ok": True, "reviewer": principal.name}

    @app.post("/api/decisions/{decision_id}/approval")
    def api_approval(decision_id: str, body: ApprovalRequest, principal: Approver):
        try:
            return execute(
                decision_id, principal.tenant, approved_by=principal.name, reject=body.reject
            )
        except LookupError as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, NotRunnable) as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/agents/{name}/value")
    def api_value(name: str, principal: Reader, weeks: int = 4) -> dict:
        _agent(name)
        if weeks < 1:
            raise HTTPException(422, "weeks must be positive")
        return measured(name, principal.tenant, weeks)

    return app
