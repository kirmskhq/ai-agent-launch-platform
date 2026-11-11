"""The exam: run an agent on past cases with known answers and decide whether it may go live.

Each run of a case ends in one of five outcomes, reported separately:

    answered_correct    the agent answered and the answer is right
    answered_wrong      the agent answered and the answer is wrong, it answered a case that
                        should have gone to a person, or its code crashed
    correct_handover    the agent handed the case to a person, and that was the right call
    unneeded_handover   the agent handed over a case it should have settled
    infra_failure       the model could not be reached; says nothing about the agent

Only answered_correct saves staff time. A correct handover is good behaviour, but the person
still does the work, so it saves nothing.

Holdout cases run several times (`exam.runs_per_case`, default 3). A case counts as correct
only if every run is correct; otherwise it takes its worst outcome. Consistency is the share of
cases where every run ended the same way.

The pass mark is applied to the lower confidence bound of answer accuracy, not to the point
estimate: 28 right out of 30 is 93%, but it does not prove the agent is 93% accurate.
"""

import ast
import copy
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from contextvars import copy_context
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_platform.cases import CaseSet, cases_path, check_case_sets, is_handover, load_cases
from agent_platform.config import models_config, platform_config
from agent_platform.economics import monthly_effect
from agent_platform.gateway import (
    GatewayError,
    exam_quota_guard,
    has_key,
    mock_model,
    track,
    use_model,
)
from agent_platform.loader import Graph, fingerprint, load_graph, load_mock_model
from agent_platform.passport import Exam, Passport, load_passport
from agent_platform.settings import ROOT, paths
from agent_platform.stats import wilson_lower_bound
from agent_platform.tools import granted

Outcome = Literal[
    "answered_correct",
    "answered_wrong",
    "correct_handover",
    "unneeded_handover",
    "infra_failure",
]
OUTCOMES: tuple[Outcome, ...] = (
    "answered_correct",
    "answered_wrong",
    "correct_handover",
    "unneeded_handover",
    "infra_failure",
)
WORST_FIRST: tuple[Outcome, ...] = (
    "infra_failure",
    "answered_wrong",
    "unneeded_handover",
    "correct_handover",
    "answered_correct",
)
CORRECT: frozenset[str] = frozenset({"answered_correct", "correct_handover"})

Matcher = Callable[[Any, Any], bool]


class ExamError(RuntimeError):
    """The exam cannot start: broken case files, missing key, unknown model."""


class Run(BaseModel):
    outcome: Outcome
    output: Any = None
    error: str | None = None
    cost: float
    latency_s: float
    model_calls: int


class CaseResult(BaseModel):
    id: str
    expected: Any
    outcome: Outcome
    consistent: bool
    runs: list[Run]

    @property
    def correct(self) -> bool:
        return self.outcome in CORRECT


class ExamReport(BaseModel):
    agent: str
    case_set: CaseSet
    model: str
    measured_on: Literal["real", "mock"]
    currency: str
    fingerprint: str
    git_sha: str | None
    finished_at: str
    duration_s: float
    runs_per_case: int
    confidence: float

    cases_total: int
    cases_measured: int
    counts: dict[Outcome, int]
    answered: int
    answer_accuracy: float | None
    accuracy_lower_bound: float | None
    answered_share: float
    consistency: float
    cost_per_case: float
    latency_p50_s: float
    latency_p95_s: float

    min_accuracy: float
    break_even_accuracy: float
    min_answered_share: float
    max_cost_per_case: float | None
    min_holdout_cases: int

    effect: dict
    effect_at_lower_bound: dict

    passed: bool
    reasons: list[str]
    cases: list[CaseResult]
    resumed_from: list[str] = Field(default_factory=list)


# ── scoring ──────────────────────────────────────────────────────────────────────────────


def canonical(value: Any) -> Any:
    """Normalise for comparison: case and spacing in text, numbers to 2 decimals, list order."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return round(float(value), 2)
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, list):
        return sorted((canonical(v) for v in value), key=repr)
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    return value


def make_matcher(exam: Exam) -> Matcher:
    def matches(output: Any, expected: Any) -> bool:
        if exam.scoring == "exact" or not isinstance(expected, dict):
            return canonical(output) == canonical(expected)
        if not isinstance(output, dict):
            return False
        keys = exam.fields or list(expected)
        return all(canonical(output.get(k)) == canonical(expected.get(k)) for k in keys)

    return matches


def classify(output: Any, expected: Any, *, error: str | None, infra: bool, matches: Matcher):
    if infra:
        return "infra_failure"
    expects_handover = is_handover(expected)
    if error is not None or output is None:
        # A crash is a defect, not a safe handover, even though production falls back to a
        # person when it happens.
        return "answered_wrong"
    if is_handover(output):
        return "correct_handover" if expects_handover else "unneeded_handover"
    if expects_handover:
        return "answered_wrong"
    return "answered_correct" if matches(output, expected) else "answered_wrong"


# ── running ──────────────────────────────────────────────────────────────────────────────


def _run_once(graph: Graph, case: dict, matches: Matcher, model: str) -> Run:
    output: Any = None
    error: str | None = None
    infra = False
    started = time.perf_counter()
    with use_model(model), track() as ledger:
        try:
            state = graph.invoke(
                copy.deepcopy(
                    {key: value for key, value in case["input"].items() if key != "output"}
                )
            )
            output = state.get("output") if isinstance(state, dict) else None
        except GatewayError as e:
            infra, error = True, f"{type(e).__name__}: {e}"
        except Exception as e:  # the agent's own failure is a failed case, not a failed exam
            error = f"{type(e).__name__}: {e}"
    if ledger.failed_calls and not infra:
        # The agent caught a model failure itself; the run measured nothing either way.
        infra = True
        error = error or "a model call failed inside the agent"
    return Run(
        outcome=classify(output, case["expected"], error=error, infra=infra, matches=matches),
        output=output,
        error=error,
        cost=ledger.cost,
        latency_s=round(time.perf_counter() - started, 3),
        model_calls=len(ledger.calls),
    )


def _settle(case: dict, runs: list[Run]) -> CaseResult:
    outcomes = {r.outcome for r in runs}
    return CaseResult(
        id=str(case["id"]),
        expected=case["expected"],
        outcome=next(o for o in WORST_FIRST if o in outcomes),
        consistent=len({json.dumps(canonical(r.output), sort_keys=True) for r in runs}) == 1
        and len(outcomes) == 1,
        runs=runs,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def accumulated_run(old: Run, new: Run) -> Run:
    return new.model_copy(
        update={
            "cost": old.cost + new.cost,
            "model_calls": old.model_calls + new.model_calls,
            "latency_s": old.latency_s + new.latency_s,
        }
    )


def validate_resume(report, passport, cases, model, case_set, runs, current_fingerprint):
    expected_kind = "mock" if models_config().is_mock(model) else "real"
    if (
        report.agent != passport.name
        or report.model != model
        or report.case_set != case_set
        or report.runs_per_case != runs
        or report.fingerprint != current_fingerprint
        or report.measured_on != expected_kind
        or report.currency != platform_config().currency
        or len(report.cases) != len(cases)
    ):
        raise ExamError("saved exam does not match the current agent, model, settings or cases")
    for saved, case in zip(report.cases, cases, strict=True):
        if (
            saved.id != str(case["id"])
            or saved.expected != case["expected"]
            or len(saved.runs) != runs
        ):
            raise ExamError("saved case IDs, labels or run slots do not match")
        matcher = make_matcher(passport.exam)
        for run in saved.runs:
            outcome = classify(
                run.output,
                case["expected"],
                error=run.error,
                infra=run.outcome == "infra_failure",
                matches=matcher,
            )
            if outcome != run.outcome:
                raise ExamError("saved run outcome disagrees with its recorded answer")


# One audited migration for reports produced before checkpointing was introduced.
# All inference, scoring, cases, model settings and other runtime sources still have to match.
PRE_CHECKPOINT_COMMIT = "38355b5605de313fe2d197e035721c1ff2f83656"


def prepare_resume(report: ExamReport, agent_dir: Path) -> ExamReport:
    passport = load_passport(agent_dir)
    current = fingerprint(agent_dir, passport)
    if report.fingerprint != current:
        overrides = {}
        for name in ("agent_platform/loader.py", "agent_platform/exam.py"):
            result = subprocess.run(
                ["git", "show", f"{PRE_CHECKPOINT_COMMIT}:{name}"],
                cwd=ROOT,
                capture_output=True,
                timeout=10,
            )
            if result.returncode:
                raise ExamError("old report requires its original commit in local Git history")
            overrides[name] = result.stdout
        old_tree = ast.parse(overrides["agent_platform/exam.py"])
        new_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        for name in (
            "canonical",
            "make_matcher",
            "classify",
            "_run_once",
            "_settle",
            "_percentile",
            "verdict",
        ):
            old = function_node(old_tree, name)
            new = function_node(new_tree, name)
            if ast.dump(old) != ast.dump(new):
                raise ExamError("evaluation logic changed; start a fresh exam")
        old_builder = function_node(old_tree, "examine")
        count_index = next(
            i
            for i, node in enumerate(old_builder.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "counts"
        )
        old_metrics = ast.Module(body=old_builder.body[count_index:], type_ignores=[])
        new_metrics = ast.Module(body=function_node(new_tree, "build_report").body, type_ignores=[])
        if ast.dump(IgnoreDuration().visit(old_metrics)) != ast.dump(
            IgnoreDuration().visit(new_metrics)
        ):
            raise ExamError("exam calculations changed; start a fresh exam")
        old_loader = ast.parse(overrides["agent_platform/loader.py"])
        new_loader = ast.parse((ROOT / "agent_platform/loader.py").read_text(encoding="utf-8"))
        for node in old_loader.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name != "fingerprint"
                and ast.dump(node) != ast.dump(function_node(new_loader, node.name))
            ):
                raise ExamError("agent loading changed; start a fresh exam")
        if fingerprint(agent_dir, passport, runtime_sources=overrides) != report.fingerprint:
            raise ExamError("saved exam predates an agent, model, policy or evaluation change")
    cases = load_cases(cases_path(agent_dir, "holdout"))
    validate_resume(
        report.model_copy(update={"fingerprint": current}),
        passport,
        cases,
        passport.model,
        "holdout",
        passport.exam.runs_per_case,
        current,
    )
    rebuilt = build_report(
        passport,
        [_settle(c, saved.runs) for c, saved in zip(cases, report.cases, strict=True)],
        case_set="holdout",
        model=passport.model,
        runs_per_case=passport.exam.runs_per_case,
        agent_fingerprint=current,
        duration_s=report.duration_s,
    )
    rebuilt.finished_at, rebuilt.git_sha = report.finished_at, report.git_sha
    rebuilt.resumed_from = list(dict.fromkeys([*report.resumed_from, report.fingerprint]))
    return rebuilt


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    node = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if node is None:
        raise ExamError(f"evaluation function {name} changed; start a fresh exam")
    return node


class IgnoreDuration(ast.NodeTransformer):
    """Compare the old and extracted calculations while allowing elapsed-time bookkeeping."""

    def visit_keyword(self, node):
        if node.arg == "duration_s":
            node.value = ast.Constant(value=0)
        return self.generic_visit(node)


def examine(
    passport: Passport,
    graph: Graph,
    cases: list[dict],
    *,
    case_set: CaseSet,
    model: str,
    runs_per_case: int,
    agent_fingerprint: str,
    workers: int = 4,
    previous: ExamReport | None = None,
    checkpoint: Path | None = None,
) -> ExamReport:
    """Run every case `runs_per_case` times and judge the result against the passport."""
    started = time.perf_counter()
    matches = make_matcher(passport.exam)
    context = copy_context()  # carries the mock responder and open ledgers into the threads
    if previous is not None:
        validate_resume(
            previous, passport, cases, model, case_set, runs_per_case, agent_fingerprint
        )
    pending = Run(outcome="infra_failure", error="not run yet", cost=0, latency_s=0, model_calls=0)
    results = [
        [r.model_copy(deep=True) for r in previous.cases[i].runs]
        if previous
        else [pending.model_copy() for _ in range(runs_per_case)]
        for i in range(len(cases))
    ]

    def save_progress():
        if checkpoint is not None:
            report = build_report(
                passport,
                [_settle(c, runs) for c, runs in zip(cases, results, strict=True)],
                case_set=case_set,
                model=model,
                runs_per_case=runs_per_case,
                agent_fingerprint=agent_fingerprint,
                duration_s=(previous.duration_s if previous else 0) + time.perf_counter() - started,
            )
            if previous:
                report.resumed_from = list(
                    dict.fromkeys([*previous.resumed_from, previous.fingerprint])
                )
            save_report(report, checkpoint)

    save_progress()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(context.copy().run, _run_once, graph, case, matches, model): (i, k)
            for i, case in enumerate(cases)
            for k, old in enumerate(results[i])
            if old.outcome == "infra_failure"
        }
        for future in as_completed(futures):
            i, k = futures[future]
            results[i][k] = accumulated_run(results[i][k], future.result())
            save_progress()
    # A provider that pauses after bursts can outlast the gateway's retries. Runs that never
    # reached the model are tried once more, one at a time, before anything is judged.
    # If a rerun still fails, the provider is down or out of quota: stop rather than send a
    # string of refused calls, which some providers punish with a longer pause.
    provider_down = False
    for case, case_runs in zip(cases, results, strict=True):
        for k, run in enumerate(case_runs):
            if run.outcome == "infra_failure" and not provider_down:
                replacement = context.copy().run(_run_once, graph, case, matches, model)
                case_runs[k] = accumulated_run(run, replacement)
                save_progress()
                provider_down = case_runs[k].outcome == "infra_failure"
    save_progress()
    results = [_settle(case, runs) for case, runs in zip(cases, results, strict=True)]

    report = build_report(
        passport,
        results,
        case_set=case_set,
        model=model,
        runs_per_case=runs_per_case,
        agent_fingerprint=agent_fingerprint,
        duration_s=(previous.duration_s if previous else 0) + time.perf_counter() - started,
    )
    if previous:
        report.resumed_from = list(dict.fromkeys([*previous.resumed_from, previous.fingerprint]))
    return report


def build_report(
    passport, results, *, case_set, model, runs_per_case, agent_fingerprint, duration_s
):
    counts = {o: sum(c.outcome == o for c in results) for o in OUTCOMES}
    measured = [c for c in results if c.outcome != "infra_failure"]
    runs = [r for c in results for r in c.runs]
    answered = counts["answered_correct"] + counts["answered_wrong"]
    accuracy = counts["answered_correct"] / answered if answered else None
    lower = (
        wilson_lower_bound(counts["answered_correct"], answered, passport.exam.confidence)
        if answered
        else None
    )
    answered_share = answered / len(measured) if measured else 0.0
    cost_per_case = mean(r.cost for r in runs) if runs else 0.0

    platform = platform_config()
    base = passport.baseline

    def effect(correct_share: float, wrong_share: float) -> dict:
        return monthly_effect(
            cases_per_month=base.cases_per_month,
            minutes_per_case=base.minutes_per_case,
            staff_cost_per_hour=base.staff_cost_per_hour,
            error_cost=passport.error_cost,
            correct_share=correct_share,
            wrong_share=wrong_share,
            model_cost_per_case=cost_per_case,
            hours_per_staff_month=platform.hours_per_staff_month,
        ).as_dict()

    n = len(measured) or 1
    point = effect(counts["answered_correct"] / n, counts["answered_wrong"] / n)
    at_lower = effect(answered_share * (lower or 0.0), answered_share * (1 - (lower or 0.0)))

    report = ExamReport(
        agent=passport.name,
        case_set=case_set,
        model=model,
        measured_on="mock" if models_config().is_mock(model) else "real",
        currency=platform.currency,
        fingerprint=agent_fingerprint,
        git_sha=_git_sha(),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        duration_s=round(duration_s, 1),
        runs_per_case=runs_per_case,
        confidence=passport.exam.confidence,
        cases_total=len(results),
        cases_measured=len(measured),
        counts=counts,
        answered=answered,
        answer_accuracy=accuracy,
        accuracy_lower_bound=lower,
        answered_share=answered_share,
        consistency=mean(c.consistent for c in measured) if measured else 0.0,
        cost_per_case=cost_per_case,
        latency_p50_s=_percentile([r.latency_s for r in runs], 0.5),
        latency_p95_s=_percentile([r.latency_s for r in runs], 0.95),
        min_accuracy=passport.exam.min_accuracy,
        break_even_accuracy=passport.break_even_accuracy,
        min_answered_share=passport.exam.min_answered_share,
        max_cost_per_case=passport.exam.max_cost_per_case,
        min_holdout_cases=passport.exam.min_holdout_cases,
        effect=point,
        effect_at_lower_bound=at_lower,
        passed=False,
        reasons=[],
        cases=results,
    )
    reasons = verdict(report)
    return report.model_copy(update={"passed": not reasons, "reasons": reasons})


def verdict(r: ExamReport) -> list[str]:
    """Why the exam is not passed. An empty list means it is."""
    reasons = []
    if r.case_set != "holdout":
        reasons.append("dev cases are for building the agent; only a holdout exam can pass")
    if r.runs_per_case != 3:
        reasons.append("acceptance requires three runs per holdout case")
    if r.confidence != 0.90:
        reasons.append("acceptance requires 90% confidence")
    if r.counts["infra_failure"]:
        reasons.append(
            f"exam incomplete: {r.counts['infra_failure']} of {r.cases_total} cases could not "
            "reach the model; rerun it"
        )
    if r.cases_total < r.min_holdout_cases:
        reasons.append(f"only {r.cases_total} cases; the passport asks for {r.min_holdout_cases}")
    if r.answered == 0:
        reasons.append("the agent answered no case itself")
    elif r.accuracy_lower_bound is not None and r.accuracy_lower_bound < r.min_accuracy:
        reasons.append(
            f"answer accuracy {r.answer_accuracy:.1%}, lower bound {r.accuracy_lower_bound:.1%} "
            f"at {r.confidence:.0%} confidence, below the required {r.min_accuracy:.1%}"
        )
    if r.answered_share < r.min_answered_share:
        reasons.append(
            f"settled {r.answered_share:.0%} of cases itself, "
            f"the passport asks for {r.min_answered_share:.0%}"
        )
    if r.max_cost_per_case is not None and r.cost_per_case > r.max_cost_per_case:
        reasons.append(
            f"model cost per case {r.cost_per_case:.4f} {r.currency} "
            f"is above the ceiling of {r.max_cost_per_case:.4f}"
        )
    if r.answered and r.effect_at_lower_bound["net"] < 0:
        reasons.append(
            f"at the accuracy lower bound the agent loses "
            f"{-r.effect_at_lower_bound['net']:,.0f} {r.currency} a month"
        )
    return reasons


def _git_sha() -> str | None:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


# ── from an agent folder ─────────────────────────────────────────────────────────────────


def run_exam(
    agent_dir: Path,
    *,
    case_set: CaseSet = "holdout",
    model: str | None = None,
    runs_per_case: int | None = None,
    workers: int = 4,
    limit: int | None = None,
    resume: Path | None = None,
    checkpoint: Path | None = None,
) -> ExamReport:
    passport = load_passport(agent_dir)
    problems = check_case_sets(agent_dir, passport.exam.min_holdout_cases)
    blocking = [p for p in problems if not p.startswith("holdout has")]
    if blocking:
        raise ExamError("; ".join(blocking))

    model = model or passport.model
    cfg = models_config()
    if model not in cfg.models:
        raise ExamError(f"model {model!r} is not in config/models.yaml")
    responder = None
    if cfg.is_mock(model):
        responder = load_mock_model(agent_dir)
        if responder is None:
            raise ExamError(f"{agent_dir.name} has no mock_model.py; it cannot run on the mock")
    elif not has_key(cfg.providers[cfg.models[model].provider]):
        provider = cfg.providers[cfg.models[model].provider]
        raise ExamError(
            f"no key for {model}: set {provider.api_key_env}, or try the agent with --model mock"
        )

    cases = load_cases(cases_path(agent_dir, case_set))[:limit]
    previous = None
    if resume is not None and resume.is_file():
        if case_set != "holdout" or limit is not None or model != passport.model:
            raise ExamError("resume requires the full holdout on the passport model")
        previous = prepare_resume(load_report(resume), agent_dir)
    graph = load_graph(agent_dir)
    runs = runs_per_case or (passport.exam.runs_per_case if case_set == "holdout" else 1)
    with (
        exam_quota_guard(),
        mock_model(responder) if responder else nullcontext(),
        granted(passport.name, passport.tools, can_write=False),
    ):
        return examine(
            passport,
            graph,
            cases,
            case_set=case_set,
            model=model,
            runs_per_case=runs,
            agent_fingerprint=fingerprint(agent_dir, passport),
            workers=workers,
            previous=previous,
            checkpoint=checkpoint,
        )


def acceptance_report_path(agent: str) -> Path:
    """The one report the registry reads to decide whether an agent may run."""
    return paths().reports / f"{agent}.json"


def default_report_path(report: ExamReport, passport_model: str) -> Path:
    if report.case_set == "dev":
        return paths().state / "exams" / f"{report.agent}.dev.json"
    if report.model != passport_model:
        return paths().reports / "trials" / f"{report.agent}@{report.model}.json"
    return acceptance_report_path(report.agent)


def save_report(report: ExamReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(report.model_dump_json(indent=1))
        temporary = stream.name
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_report(path: Path) -> ExamReport:
    return ExamReport.model_validate_json(path.read_text(encoding="utf-8"))
