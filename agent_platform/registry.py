"""The registry: passport + acceptance report -> what the agent is allowed to do.

Status is derived, never typed in by hand:

    draft         the passport says lifecycle: draft; being built, never runs
    retired       the passport says lifecycle: retired; switched off
    no_exam       live, but there is no acceptance report yet
    not_accepted  live, but the exam failed, ran on another model or on the mock, is stale
                  (the agent changed since), or the passport itself is invalid
    running       live and accepted on the passport's model; the only status that runs

The acceptance report is reports/<agent>.json. CI writes it; the server only reads it.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from agent_platform.cases import check_case_sets
from agent_platform.config import models_config
from agent_platform.exam import ExamReport, acceptance_report_path, load_report
from agent_platform.loader import check_code, fingerprint
from agent_platform.passport import NAME_PATTERN, Passport, PassportError, load_passport
from agent_platform.settings import paths

Status = Literal["draft", "retired", "no_exam", "not_accepted", "running"]
STATUS_LABELS: dict[Status, str] = {
    "draft": "draft",
    "retired": "retired",
    "no_exam": "no exam",
    "not_accepted": "not accepted",
    "running": "running",
}


@dataclass
class AgentRecord:
    name: str
    path: Path
    passport: Passport | None
    report: ExamReport | None
    status: Status
    reasons: list[str] = field(default_factory=list)

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def runs(self) -> bool:
        return self.status == "running"


def discover(agents_dir: Path | None = None) -> list[Path]:
    root = agents_dir or paths().agents
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_")) and (p / "agent.yaml").is_file()
    )


def check_agent(agent_dir: Path) -> list[str]:
    """Everything wrong with an agent folder. An empty list means it is ready for an exam."""
    try:
        passport = load_passport(agent_dir)
    except PassportError as e:
        return [f"agent.yaml: {p}" for p in e.problems]
    problems = []
    if passport.name != agent_dir.name:
        problems.append(f"name {passport.name!r} does not match the folder {agent_dir.name!r}")
    cfg = models_config()
    if passport.model not in cfg.models:
        problems.append(
            f"model {passport.model!r} is not in config/models.yaml "
            f"(known: {', '.join(sorted(cfg.models))})"
        )
    problems += check_code(agent_dir)
    problems += check_case_sets(agent_dir, passport.exam.min_holdout_cases)
    return problems


def _read_report(name: str) -> tuple[ExamReport | None, str | None]:
    path = acceptance_report_path(name)
    if not path.is_file():
        return None, None
    try:
        return load_report(path), None
    except (ValidationError, ValueError) as e:
        return None, f"acceptance report is unreadable: {str(e).splitlines()[0]}"


def derive_status(
    passport: Passport, report: ExamReport | None, current_fingerprint: str
) -> tuple[Status, list[str]]:
    if passport.lifecycle == "retired":
        return "retired", ["retired in the passport"]
    if passport.lifecycle == "draft":
        return "draft", ["draft in the passport"]
    if report is None:
        return "no_exam", ["no acceptance exam yet"]
    if report.case_set != "holdout":
        return "not_accepted", ["the report is from dev cases, not the holdout"]
    if report.measured_on != "real" or models_config().is_mock(passport.model):
        return "not_accepted", ["measured on the mock model, not a real one"]
    if report.model != passport.model:
        return "not_accepted", [
            f"the exam ran on {report.model}, the passport names {passport.model}"
        ]
    if report.fingerprint != current_fingerprint:
        return "not_accepted", ["the agent changed after its exam; rerun the exam"]
    if not report.passed:
        return "not_accepted", list(report.reasons) or ["the exam was not passed"]
    return "running", []


def record(agent_dir: Path) -> AgentRecord:
    try:
        passport = load_passport(agent_dir)
    except PassportError as e:
        return AgentRecord(
            agent_dir.name, agent_dir, None, None, "not_accepted", [f"passport: {e}"]
        )
    report, unreadable = _read_report(passport.name)
    if unreadable:
        return AgentRecord(passport.name, agent_dir, passport, None, "not_accepted", [unreadable])
    status, reasons = derive_status(passport, report, fingerprint(agent_dir, passport))
    return AgentRecord(passport.name, agent_dir, passport, report, status, reasons)


def records(agents_dir: Path | None = None) -> list[AgentRecord]:
    return [record(d) for d in discover(agents_dir)]


def find(name: str) -> AgentRecord | None:
    """The agent called `name`, or None. Names that are not valid agent names never touch disk."""
    if not re.fullmatch(NAME_PATTERN, name):
        return None
    path = paths().agents / name
    if not (path / "agent.yaml").is_file():
        return None
    return record(path)
