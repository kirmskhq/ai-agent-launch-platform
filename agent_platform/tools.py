"""The tool registry (config/tools.yaml) and the one way agents call tools.

Access is decided by policy, not by agent code:
- the passport lists the tools an agent may call; anything else is refused;
- a tool that writes needs autonomy above "advise" in the passport;
- while an exam runs, no tool that writes can run, whatever the passport says.

Every call is checked again at run time, so a passport check that was skipped cannot open a
door. Refusals raise `ToolRefused`; they are bugs in the agent, not answers.
"""

import importlib
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from agent_platform.settings import paths


class ToolRefused(PermissionError):
    pass


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    function: str
    writes: bool
    description: str


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tools: dict[str, ToolSpec] = {}


@lru_cache(maxsize=8)
def _load(path: str, _mtime: int) -> ToolsConfig:
    with Path(path).open(encoding="utf-8") as f:
        return ToolsConfig.model_validate(yaml.safe_load(f) or {})


def tools_config() -> ToolsConfig:
    path = paths().config / "tools.yaml"
    if not path.is_file():
        return ToolsConfig()
    return _load(str(path), path.stat().st_mtime_ns)


@dataclass(frozen=True)
class ToolCall:
    tool: str
    latency_s: float
    error: str | None = None


@dataclass
class Grant:
    agent: str
    tools: frozenset[str]
    can_write: bool
    calls: list[ToolCall] = field(default_factory=list)


_grant: ContextVar[Grant | None] = ContextVar("tool_grant", default=None)


@contextmanager
def granted(agent: str, tools: list[str], *, can_write: bool) -> Iterator[Grant]:
    """Open the agent's tools for the block. The runner and the exam set this, never agents."""
    grant = Grant(agent, frozenset(tools), can_write)
    token = _grant.set(grant)
    try:
        yield grant
    finally:
        _grant.reset(token)


def check(name: str) -> ToolSpec:
    grant = _grant.get()
    spec = tools_config().tools.get(name)
    if spec is None:
        raise ToolRefused(f"no tool called {name!r} in config/tools.yaml")
    if grant is None:
        raise ToolRefused(f"{name}: tools can only be called while the platform runs an agent")
    if name not in grant.tools:
        raise ToolRefused(f"{name} is not in the passport of {grant.agent}")
    if spec.writes and not grant.can_write:
        raise ToolRefused(f"{name} writes, and {grant.agent} may not write here")
    return spec


def _function(spec: ToolSpec) -> Callable[..., Any]:
    module, _, attr = spec.function.partition(":")
    return getattr(importlib.import_module(module), attr)


def call(name: str, /, **arguments: Any) -> Any:
    spec = check(name)
    grant = _grant.get()
    started = time.perf_counter()
    try:
        result = _function(spec)(**arguments)
    except Exception as e:
        if grant is not None:
            grant.calls.append(ToolCall(name, time.perf_counter() - started, str(e)))
        raise
    if grant is not None:
        grant.calls.append(ToolCall(name, time.perf_counter() - started))
    return result


def passport_problems(tools: list[str], autonomy: str) -> list[str]:
    known = tools_config().tools
    problems = [f"tool {t!r} is not in config/tools.yaml" for t in tools if t not in known]
    writers = [t for t in tools if t in known and known[t].writes]
    if writers and autonomy == "advise":
        problems.append(f"tools that write ({', '.join(writers)}) need autonomy above advise")
    return problems
