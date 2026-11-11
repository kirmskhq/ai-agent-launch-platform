"""The agent service: catalogue pages, a JSON API, and the endpoints that run agents.

    GET  /                           catalogue
    GET  /agents/{name}              one agent
    GET  /api/agents                 catalogue as JSON
    GET  /api/agents/{name}          one agent as JSON
    POST /api/agents/{name}/invoke   one call: {"input": {...}}             bearer token
    POST /api/agents/{name}/run      the agent's scheduled job, right now     bearer token
    GET  /healthz                    liveness

Reading is open; anything that runs an agent needs `Authorization: Bearer <AGENTS_API_TOKEN>`.
If AGENTS_API_TOKEN is not set, those endpoints refuse every request rather than run open.

The scheduler starts with the service when AGENTS_SCHEDULER=1.
"""

import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from agent_platform.catalogue import render_agent, render_index
from agent_platform.registry import AgentRecord, find, records
from agent_platform.runner import NotRunnable, invoke, run_job
from agent_platform.scheduler import Scheduler
from agent_platform.settings import load_dotenv

log = logging.getLogger("agent_platform.service")
_bearer = HTTPBearer(auto_error=False)


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    expected = os.environ.get("AGENTS_API_TOKEN", "")
    if not expected:
        raise HTTPException(503, "AGENTS_API_TOKEN is not set; running agents is disabled")
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode(), expected.encode()
    ):
        raise HTTPException(401, "missing or wrong bearer token")


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
    load_dotenv()
    start_scheduler = os.environ.get("AGENTS_SCHEDULER") == "1" if scheduler is None else scheduler

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sched = Scheduler()
        if start_scheduler:
            sched.start()
            log.info("scheduler started (UTC)")
        yield
        sched.stop()

    app = FastAPI(title="Agent launch platform", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_index(records())

    @app.get("/agents/{name}", response_class=HTMLResponse)
    def agent_page(name: str) -> str:
        return render_agent(_agent(name))

    @app.get("/api/agents")
    def api_agents() -> list[dict]:
        return [_summary(a) for a in records()]

    @app.get("/api/agents/{name}")
    def api_agent(name: str) -> dict:
        return _summary(_agent(name))

    @app.post("/api/agents/{name}/invoke", dependencies=[Depends(require_token)])
    async def api_invoke(name: str, body: InvokeRequest) -> dict:
        _agent(name)
        try:
            result = await run_in_threadpool(invoke, name, body.input)
        except NotRunnable as e:
            raise HTTPException(409, str(e)) from e
        return {
            "output": result.output,
            "fell_back_to_person": result.fell_back,
            "cost": round(result.cost, 6),
            "model_calls": result.model_calls,
            "latency_s": round(result.latency_s, 3),
        }

    @app.post("/api/agents/{name}/run", dependencies=[Depends(require_token)])
    async def api_run(name: str) -> dict:
        _agent(name)
        try:
            return await run_in_threadpool(run_job, name, trigger="api")
        except NotRunnable as e:
            raise HTTPException(409, str(e)) from e

    return app
