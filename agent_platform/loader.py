"""Loading an agent's code from its folder.

An agent folder holds:
    agent.yaml       the passport
    graph.py         build_graph() -> an object with .invoke(state: dict) -> dict
                     (a compiled LangGraph graph, or anything with the same method).
                     The answer goes in state["output"]; a handover to a person is
                     {"handover": true, "reason": "..."}.
    mock_model.py    optional: respond(messages) -> str, a stand-in for the model so the agent
                     can be exercised without a key. Results on it never count for acceptance.
    job.py           optional: collect() -> list of inputs, for scheduled runs.
    evals/dev.jsonl, evals/holdout.jsonl
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from agent_platform.cases import cases_path
from agent_platform.gateway import MockModel
from agent_platform.passport import Passport


class Graph(Protocol):
    def invoke(self, state: dict, /) -> dict: ...


def _import(agent_dir: Path, filename: str) -> ModuleType:
    path = agent_dir / filename
    module_name = f"agent_code.{agent_dir.name.replace('-', '_')}.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_graph(agent_dir: Path) -> Graph:
    module = _import(agent_dir, "graph.py")
    if not hasattr(module, "build_graph"):
        raise ImportError(f"{agent_dir.name}/graph.py has no build_graph()")
    return module.build_graph()


def load_mock_model(agent_dir: Path) -> MockModel | None:
    if not (agent_dir / "mock_model.py").is_file():
        return None
    return _import(agent_dir, "mock_model.py").respond


def load_job(agent_dir: Path) -> Any | None:
    if not (agent_dir / "job.py").is_file():
        return None
    return _import(agent_dir, "job.py")


def check_code(agent_dir: Path) -> list[str]:
    problems = []
    graph_file = agent_dir / "graph.py"
    if not graph_file.is_file():
        problems.append("no graph.py")
    elif "def build_graph" not in graph_file.read_text(encoding="utf-8"):
        problems.append("graph.py has no build_graph()")
    mock_file = agent_dir / "mock_model.py"
    if mock_file.is_file() and "def respond" not in mock_file.read_text(encoding="utf-8"):
        problems.append("mock_model.py has no respond(messages)")
    job_file = agent_dir / "job.py"
    if job_file.is_file() and "def collect" not in job_file.read_text(encoding="utf-8"):
        problems.append("job.py has no collect()")
    return problems


def fingerprint(agent_dir: Path, passport: Passport) -> str:
    """A hash of everything that decides the exam result.

    The agent's code, its holdout cases and the passport fields the exam depends on. If any of
    them changes after the exam, the exam no longer describes the agent and has to be rerun.
    Schedule, title and lifecycle are left out: changing them does not change answers.
    """
    h = hashlib.sha256()
    for path in sorted(agent_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(path.relative_to(agent_dir).as_posix().encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    holdout = cases_path(agent_dir, "holdout")
    if holdout.is_file():
        h.update(holdout.read_bytes().replace(b"\r\n", b"\n"))
    exam_inputs = passport.model_dump(include={"model", "exam", "baseline", "error_cost"})
    h.update(json.dumps(exam_inputs, sort_keys=True).encode())
    return h.hexdigest()[:16]
