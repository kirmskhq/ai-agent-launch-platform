"""Catalogue pages: one table of all agents and one page per agent."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent_platform.config import platform_config
from agent_platform.readiness import Readiness
from agent_platform.registry import AgentRecord
from agent_platform.runner import recent_runs

TEMPLATES = Path(__file__).parent / "templates"

OUTCOME_LABELS = {
    "answered_correct": "answered, correct",
    "answered_wrong": "answered, wrong",
    "correct_handover": "handed over, rightly",
    "unneeded_handover": "handed over, needlessly",
    "infra_failure": "model unreachable",
}


def _money(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "-"
    return f"{x:,.{digits}f} {platform_config().currency}"


def _pct(x: float | None, digits: int = 0) -> str:
    return "-" if x is None else f"{x:.{digits}%}"


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(), trim_blocks=True
    )
    env.filters["money"] = _money
    env.filters["pct"] = _pct
    env.globals["outcome_labels"] = OUTCOME_LABELS
    return env


def summary(agents: list[AgentRecord]) -> dict:
    running = [a for a in agents if a.runs and a.report]
    return {
        "total": len(agents),
        "running": len(running),
        "projected_net": sum(a.report.effect_at_lower_bound["net"] for a in running),
    }


def render_index(agents: list[AgentRecord], proposals: list[Readiness]) -> str:
    return (
        _env()
        .get_template("index.html")
        .render(agents=agents, proposals=proposals, summary=summary(agents))
    )


def render_proposal(readiness: Readiness) -> str:
    return _env().get_template("proposal.html").render(rd=readiness, p=readiness.proposal)


def render_agent(agent: AgentRecord) -> str:
    return (
        _env()
        .get_template("agent.html")
        .render(a=agent, p=agent.passport, r=agent.report, runs=recent_runs(agent.name))
    )
