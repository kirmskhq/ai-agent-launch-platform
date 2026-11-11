from collections import Counter

import pytest

from agent_platform import exam
from agent_platform.gateway import GatewayError
from agent_platform.passport import load_passport


class Graph:
    def __init__(self, broken=False):
        self.broken = broken
        self.calls = Counter()

    def invoke(self, payload):
        key = payload["key"]
        self.calls[key] += 1
        if key == "offline" and self.broken:
            raise GatewayError("provider unavailable")
        return {"output": {"category": "meals" if key == "wrong" else "travel"}}


CASES = [
    {"id": name, "input": {"key": name}, "expected": {"category": "travel"}}
    for name in ("correct", "wrong", "offline")
]


def run(workspace, graph, previous=None, checkpoint=None, cases=None, fingerprint="same"):
    return exam.examine(
        load_passport(workspace / "agents" / "expense-coder"),
        graph,
        cases or CASES,
        case_set="holdout",
        model="mock",
        runs_per_case=3,
        agent_fingerprint=fingerprint,
        workers=1,
        previous=previous,
        checkpoint=checkpoint,
    )


def test_resume_preserves_good_and_wrong_answers_and_accumulates_cost(workspace):
    original = run(workspace, Graph(broken=True))
    original.cases[2].runs[0].cost = 0.25
    # Two slots of this case already completed successfully before the interruption.
    for index in (1, 2):
        original.cases[2].runs[index] = original.cases[0].runs[index].model_copy(deep=True)
    graph = Graph()
    path = workspace / "checkpoint.json"
    resumed = run(workspace, graph, previous=original, checkpoint=path)
    assert graph.calls == {"offline": 1}
    assert resumed.cases[0].runs == original.cases[0].runs
    assert resumed.cases[1].runs == original.cases[1].runs
    assert resumed.counts["answered_wrong"] == 1
    assert resumed.counts["infra_failure"] == 0
    assert resumed.cases[2].runs[0].cost == 0.25
    assert all(len(case.runs) == 3 for case in exam.load_report(path).cases)


@pytest.mark.parametrize("change", ["fingerprint", "model", "cases", "slots", "outcome"])
def test_resume_rejects_changed_or_inconsistent_evidence(workspace, change):
    previous = run(workspace, Graph(broken=True))
    if change == "fingerprint":
        previous.fingerprint = "different"
    elif change == "model":
        previous.model = "another-model"
    elif change == "cases":
        previous.cases[0].expected = {"category": "equipment"}
    elif change == "slots":
        previous.cases[0].runs.pop()
    else:
        previous.cases[1].runs[0].outcome = "answered_correct"
    graph = Graph()
    with pytest.raises(exam.ExamError):
        run(workspace, graph, previous=previous)
    assert not graph.calls


def test_checkpoint_keeps_completed_slot_when_execution_is_interrupted(workspace, monkeypatch):
    path = workspace / "checkpoint.json"
    real_save = exam.save_report
    saves = 0

    def interrupted_save(report, target):
        nonlocal saves
        real_save(report, target)
        saves += 1
        if saves == 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(exam, "save_report", interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        run(workspace, Graph(), checkpoint=path)
    checkpoint = exam.load_report(path)
    completed = sum(r.outcome != "infra_failure" for c in checkpoint.cases for r in c.runs)
    assert completed == 1
    assert not checkpoint.passed
    monkeypatch.setattr(exam, "save_report", real_save)
    graph = Graph()
    resumed = run(workspace, graph, previous=checkpoint, checkpoint=path)
    assert sum(graph.calls.values()) == 8
    assert resumed.counts["infra_failure"] == 0
