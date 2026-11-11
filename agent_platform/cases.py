"""Exam cases: evals/dev.jsonl for building the agent, evals/holdout.jsonl for acceptance.

The two sets must not share cases. Builders look at dev cases all day and tune prompts and
rules on them; a score on those cases says little about new ones. Holdout cases are kept apart
and only used to decide acceptance.

One case per line: {"id": "...", "input": {...}, "expected": {...}}. A case where the right
move is to hand over to a person has "expected": {"handover": true}.
"""

import json
from pathlib import Path
from typing import Any, Literal

CaseSet = Literal["dev", "holdout"]
CASE_SETS: tuple[CaseSet, ...] = ("dev", "holdout")


def cases_path(agent_dir: Path, case_set: CaseSet) -> Path:
    return agent_dir / "evals" / f"{case_set}.jsonl"


def is_handover(value: Any) -> bool:
    return isinstance(value, dict) and value.get("handover") is True


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name} line {n}: not JSON ({e.msg})") from e
            if not isinstance(case, dict):
                raise ValueError(f"{path.name} line {n}: expected a JSON object")
            missing = {"id", "input", "expected"} - case.keys()
            if missing:
                raise ValueError(f"{path.name} line {n}: missing {', '.join(sorted(missing))}")
            if not isinstance(case["id"], str) or not case["id"].strip():
                raise ValueError(f"{path.name} line {n}: id must be a nonempty string")
            if not isinstance(case["input"], dict):
                raise ValueError(f"{path.name} line {n}: input must be an object")
            cases.append(case)
    ids = [c["id"] for c in cases]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"{path.name}: repeated case ids: {', '.join(map(str, repeated))}")
    return cases


def _key(case: dict) -> str:
    return json.dumps(case["input"], sort_keys=True)


def check_case_sets(agent_dir: Path, min_holdout: int) -> list[str]:
    """Problems with the agent's case files; an empty list means they are fine."""
    problems: list[str] = []
    loaded: dict[str, list[dict]] = {}
    for case_set in CASE_SETS:
        path = cases_path(agent_dir, case_set)
        if not path.is_file():
            problems.append(f"no evals/{case_set}.jsonl")
            continue
        try:
            loaded[case_set] = load_cases(path)
        except ValueError as e:
            problems.append(str(e))
    if "holdout" in loaded and len(loaded["holdout"]) < min_holdout:
        problems.append(
            f"holdout has {len(loaded['holdout'])} cases, the passport asks for {min_holdout}"
        )
    if "dev" in loaded and "holdout" in loaded:
        dev_ids = {c["id"] for c in loaded["dev"]}
        dev_inputs = {_key(c) for c in loaded["dev"]}
        shared_ids = sorted(str(c["id"]) for c in loaded["holdout"] if c["id"] in dev_ids)
        shared_inputs = sorted(str(c["id"]) for c in loaded["holdout"] if _key(c) in dev_inputs)
        if shared_ids:
            problems.append(f"case ids in both dev and holdout: {', '.join(shared_ids)}")
        if shared_inputs:
            problems.append(
                f"holdout cases with the same input as a dev case: {', '.join(shared_inputs)}"
            )
    return problems
