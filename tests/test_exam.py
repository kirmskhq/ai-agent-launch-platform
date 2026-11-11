import itertools
import threading

import pytest

from agent_platform.exam import (
    ExamError,
    classify,
    examine,
    make_matcher,
    run_exam,
)
from agent_platform.gateway import GatewayError, ModelUnavailable, chat
from agent_platform.loader import fingerprint
from agent_platform.passport import Exam, load_passport

from .conftest import make_agent

matches = make_matcher(Exam(min_accuracy=0.5))
ANSWER = {"category": "travel"}
HANDOVER = {"handover": True}


@pytest.mark.parametrize(
    ("output", "expected", "error", "infra", "outcome"),
    [
        ({"category": "Travel "}, ANSWER, None, False, "answered_correct"),
        ({"category": "meals"}, ANSWER, None, False, "answered_wrong"),
        ({"category": "travel"}, HANDOVER, None, False, "answered_wrong"),
        (HANDOVER | {"reason": "unsure"}, HANDOVER, None, False, "correct_handover"),
        (HANDOVER, ANSWER, None, False, "unneeded_handover"),
        (None, ANSWER, "ValueError: boom", False, "answered_wrong"),
        (None, ANSWER, None, False, "answered_wrong"),
        (None, ANSWER, "ModelUnavailable", True, "infra_failure"),
    ],
)
def test_classify(output, expected, error, infra, outcome):
    assert classify(output, expected, error=error, infra=infra, matches=matches) == outcome


def test_matcher_ignores_extra_fields_and_rounds_money():
    m = make_matcher(Exam(min_accuracy=0.5))
    assert m({"amount": 10.001, "note": "x"}, {"amount": 10.0})
    assert not m({"amount": 10.2}, {"amount": 10.0})
    assert m({"ids": ["b", "a"]}, {"ids": ["a", "b"]})
    exact = make_matcher(Exam(min_accuracy=0.5, scoring="exact"))
    assert not exact({"amount": 10.0, "note": "x"}, {"amount": 10.0})


class ScriptedGraph:
    """Answers from a per-case script; each run of a case takes the next scripted answer."""

    def __init__(self, script: dict[str, list]):
        self._iters = {k: itertools.cycle(v) for k, v in script.items()}
        self._lock = threading.Lock()

    def invoke(self, state: dict) -> dict:
        with self._lock:
            step = next(self._iters[state["key"]])
        if isinstance(step, Exception):
            raise step
        if step == "model-call":
            return {"output": {"category": chat("hi")}}
        return {"output": step}


def run(passport, script: dict[str, list], expected: dict[str, dict], runs=3, case_set="holdout"):
    cases = [{"id": k, "input": {"key": k}, "expected": expected[k]} for k in script]
    return examine(
        passport,
        ScriptedGraph(script),
        cases,
        case_set=case_set,
        model="mock",
        runs_per_case=runs,
        agent_fingerprint="x",
        workers=4,
    )


@pytest.fixture
def passport(workspace):
    return load_passport(workspace / "agents" / "expense-coder")


def test_a_case_is_correct_only_if_every_run_is(passport):
    r = run(
        passport,
        {"steady": [ANSWER], "flaky": [ANSWER, {"category": "meals"}, ANSWER]},
        {"steady": ANSWER, "flaky": ANSWER},
    )
    by_id = {c.id: c for c in r.cases}
    assert by_id["steady"].outcome == "answered_correct" and by_id["steady"].consistent
    assert by_id["flaky"].outcome == "answered_wrong" and not by_id["flaky"].consistent
    assert r.consistency == 0.5
    assert r.counts["answered_correct"] == 1 and r.counts["answered_wrong"] == 1


def test_flaky_handover_takes_the_worse_outcome(passport):
    r = run(passport, {"c": [ANSWER, HANDOVER, ANSWER]}, {"c": ANSWER})
    assert r.cases[0].outcome == "unneeded_handover"


def test_correct_handover_saves_no_time(passport):
    r = run(passport, {"a": [HANDOVER], "b": [HANDOVER]}, {"a": HANDOVER, "b": HANDOVER})
    assert r.counts["correct_handover"] == 2
    assert r.answered == 0
    assert r.effect["staff_saving"] == 0
    assert "the agent answered no case itself" in r.reasons


def test_five_outcomes_are_reported_separately(passport):
    script = {
        "ok": [ANSWER],
        "wrong": [{"category": "meals"}],
        "rightly-passed": [HANDOVER],
        "needlessly-passed": [HANDOVER],
        "down": [ModelUnavailable("provider down")],
    }
    expected = {k: ANSWER for k in script} | {"rightly-passed": HANDOVER}
    r = run(passport, script, expected)
    assert r.counts == {
        "answered_correct": 1,
        "answered_wrong": 1,
        "correct_handover": 1,
        "unneeded_handover": 1,
        "infra_failure": 1,
    }
    assert r.cases_measured == 4
    assert r.answered_share == 0.5
    assert any(reason.startswith("exam incomplete") for reason in r.reasons)


def test_runs_that_never_reached_the_model_are_tried_again(passport):
    r = run(passport, {"c": [ModelUnavailable("paused"), ANSWER, ANSWER, ANSWER]}, {"c": ANSWER})
    assert r.cases[0].outcome == "answered_correct"
    assert r.counts["infra_failure"] == 0


def test_rerun_stops_when_the_provider_stays_down(passport):
    down = ModelUnavailable("quota used up")
    script = {"a": [down], "b": [down], "c": [down]}
    calls = []

    class Counting(ScriptedGraph):
        def invoke(self, state):
            calls.append(state["key"])
            return super().invoke(state)

    cases = [{"id": k, "input": {"key": k}, "expected": ANSWER} for k in script]
    r = examine(
        passport,
        Counting(script),
        cases,
        case_set="holdout",
        model="mock",
        runs_per_case=1,
        agent_fingerprint="x",
    )
    assert r.counts["infra_failure"] == 3
    assert len(calls) == 4  # three runs, then one rerun that also failed


def test_agent_crash_is_a_wrong_answer(passport):
    r = run(passport, {"c": [ValueError("bad input")]}, {"c": ANSWER}, runs=1)
    assert r.cases[0].outcome == "answered_wrong"
    assert "ValueError" in r.cases[0].runs[0].error


def test_swallowed_model_failure_is_infrastructure(passport):
    class Swallows:
        def invoke(self, state):
            try:
                chat("hi")  # the mock has no responder here: GatewayError
            except GatewayError:
                return {"output": HANDOVER}
            return {"output": ANSWER}

    cases = [{"id": "c", "input": {}, "expected": ANSWER}]
    r = examine(
        passport,
        Swallows(),
        cases,
        case_set="holdout",
        model="mock",
        runs_per_case=1,
        agent_fingerprint="x",
    )
    assert r.cases[0].outcome == "infra_failure"


def test_dev_cases_never_pass(passport):
    script = {f"c{i}": [ANSWER] for i in range(40)}
    r = run(passport, script, dict.fromkeys(script, ANSWER), case_set="dev")
    assert not r.passed
    assert any("dev cases" in reason for reason in r.reasons)


def test_pass_uses_the_lower_bound(workspace):
    """8 of 9 answers right is 88.9%, above the 75% bar, but the lower bound is only 69%."""
    report = run_exam(workspace / "agents" / "expense-coder", model="mock")
    assert report.counts == {
        "answered_correct": 8,
        "answered_wrong": 1,
        "correct_handover": 2,
        "unneeded_handover": 1,
        "infra_failure": 0,
    }
    assert report.answer_accuracy == pytest.approx(8 / 9)
    assert report.accuracy_lower_bound == pytest.approx(0.6916, abs=1e-3)
    assert not report.passed
    assert any("lower bound 69.2%" in reason for reason in report.reasons)
    assert report.runs_per_case == 3
    assert report.consistency == 1.0
    assert report.measured_on == "mock"


def test_projection_at_lower_bound_is_lower(workspace):
    report = run_exam(workspace / "agents" / "expense-coder", model="mock")
    assert report.effect_at_lower_bound["net"] < report.effect["net"]
    assert report.effect["handed_over"] == pytest.approx(2000 * 3 / 12)


def test_real_model_exam_passes(workspace, real_model):
    agent = make_agent(workspace, "expense-real", model="test-real")
    report = run_exam(agent)
    assert report.passed, report.reasons
    assert report.measured_on == "real"
    assert report.counts["answered_correct"] == 10
    assert report.cost_per_case > 0
    assert report.effect["model_cost"] == pytest.approx(2000 * report.cost_per_case, abs=1e-3)


def test_exam_needs_a_key(workspace):
    agent = make_agent(workspace, "expense-real", model="test-real")
    with pytest.raises(ExamError, match="TESTCLOUD_KEY"):
        run_exam(agent)


def test_exam_needs_a_mock_responder_for_the_mock(workspace):
    agent = make_agent(workspace, "no-mock")
    (agent / "mock_model.py").unlink()
    with pytest.raises(ExamError, match="no mock_model.py"):
        run_exam(agent, model="mock")


def test_holdout_overlapping_dev_blocks_the_exam(workspace):
    agent = workspace / "agents" / "expense-coder"
    holdout = (agent / "evals" / "holdout.jsonl").read_text(encoding="utf-8")
    with (agent / "evals" / "dev.jsonl").open("a", encoding="utf-8") as f:
        f.write(holdout.splitlines()[0].replace('"h01"', '"d99"') + "\n")
    with pytest.raises(ExamError, match="same input as a dev case"):
        run_exam(agent, model="mock")


def test_fingerprint_follows_code_and_holdout(workspace):
    agent = workspace / "agents" / "expense-coder"
    passport = load_passport(agent)
    before = fingerprint(agent, passport)
    with (agent / "evals" / "dev.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "d6", "input": {"description": "x", "amount": 1}, "expected": {}}\n')
    assert fingerprint(agent, passport) == before  # dev cases do not decide acceptance
    graph = agent / "graph.py"
    graph.write_text(graph.read_text(encoding="utf-8") + "\n# tweak\n", encoding="utf-8")
    assert fingerprint(agent, passport) != before
