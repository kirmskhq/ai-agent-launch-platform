"""The readiness checkpoint: is this process ready for an agent, before anyone writes code?

A department writes `proposal.yaml` and runs a reality test (`readiness/reality_test.jsonl`):

- a process map: steps, decisions with their written rules, inputs, outputs, systems and
  exceptions, signed off by the process owner by name and date;
- a reality test: two staff members each take the same past cases (30 or more) through the
  written process alone. Coverage is the share of cases the written rules settled for both of
  them. Agreement is the share where both reached the same answer. If two trained people agree
  on 88% of cases, nobody can honestly ask an agent for 95%: agreement is the ceiling for the
  agent's accuracy threshold;
- the cost of one error and whether it can be undone, which caps the agent's autonomy;
- a baseline and target for one business metric, and one guardrail metric that must not
  get worse.

Every check is plain pass or fail; there is no weighted score that lets a strong point hide a
missing one. The platform then recommends one outcome:

    build         every check passed
    fix_process   something is missing or the process is too loose to automate yet
    decline       the numbers cannot work even with a perfect process

The reviewer writes the decision and its reason in the proposal. The written decision can be
stricter than the recommendation, never looser: "build" is only accepted when the platform
also says build. No agent code may exist in the folder until then.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent_platform.config import ReadinessRules, platform_config
from agent_platform.economics import break_even_accuracy
from agent_platform.exam import canonical
from agent_platform.passport import Baseline, Metric, Passport

Tier = Literal["advise", "act_after_approval", "act_alone"]
TIERS: tuple[Tier, ...] = ("advise", "act_after_approval", "act_alone")
Outcome = Literal["build", "fix_process", "decline"]

PROPOSAL_FILE = "proposal.yaml"
REALITY_TEST_FILE = Path("readiness") / "reality_test.jsonl"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Person(_Strict):
    name: str = Field(min_length=3)
    role: str = Field(min_length=2)


class Step(_Strict):
    id: str
    does: str = Field(min_length=5, description="What happens in this step, plainly")
    by: str = Field(description="Role or system that does it")
    inputs: list[str] = []
    outputs: list[str] = []
    systems: list[str] = []
    decision: str | None = Field(default=None, description="Id of the decision taken here")


class Decision(_Strict):
    id: str
    question: str = Field(min_length=5)
    rule: str = Field(min_length=10, description="The written rule a person applies")


class ProcessException(_Strict):
    when: str = Field(min_length=5)
    then: str = Field(min_length=5)


class SignOff(_Strict):
    by: str
    signed_on: date


class ProcessMap(_Strict):
    steps: list[Step]
    decisions: list[Decision] = []
    exceptions: list[ProcessException] = []
    signed_off: SignOff | None = None

    @property
    def inputs(self) -> list[str]:
        return sorted({i for s in self.steps for i in s.inputs})

    @property
    def outputs(self) -> list[str]:
        return sorted({o for s in self.steps for o in s.outputs})

    @property
    def systems(self) -> list[str]:
        return sorted({x for s in self.steps for x in s.systems})


class RealityTest(_Strict):
    staff: list[str] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _two_people(self) -> "RealityTest":
        if len({s.casefold() for s in self.staff}) != 2:
            raise ValueError("the reality test needs two different staff members")
        return self


class ErrorCost(_Strict):
    cost: float = Field(ge=0, description="Cost of one wrong answer that goes out")
    reversible: bool
    how_undone: str | None = None

    @model_validator(mode="after")
    def _explain_undo(self) -> "ErrorCost":
        if self.reversible and not self.how_undone:
            raise ValueError("a reversible error needs how_undone: who undoes it, and how")
        return self


class Guardrail(_Strict):
    name: str = Field(min_length=3)
    unit: str
    current: float
    must_stay: Literal["below", "above"]
    limit: float

    @model_validator(mode="after")
    def _currently_inside(self) -> "Guardrail":
        inside = (
            self.current < self.limit if self.must_stay == "below" else self.current > self.limit
        )
        if not inside:
            raise ValueError("the guardrail is already breached today; fix that first")
        return self


class WrittenDecision(_Strict):
    outcome: Outcome
    reason: str = Field(min_length=20)
    by: str
    decided_on: date


class Proposal(_Strict):
    name: str
    title: str
    department: str
    owner: Person
    problem: str = Field(min_length=20, description="What goes wrong today, plainly")
    process: ProcessMap
    reality_test: RealityTest
    error: ErrorCost
    autonomy: Tier = Field(description="The autonomy the department asks for")
    baseline: Baseline
    metric: Metric
    guardrail: Guardrail
    decision: WrittenDecision | None = None


# ── results ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Check:
    id: str
    label: str
    passed: bool
    detail: str


@dataclass
class RealityResult:
    cases: int
    coverage: float
    agreement: float
    disagreements: list[str]
    unsettled: list[str]


@dataclass
class Readiness:
    name: str
    proposal: Proposal | None
    problems: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    reality: RealityResult | None = None
    max_autonomy: Tier | None = None
    recommended: Outcome | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def decision(self) -> WrittenDecision | None:
        return self.proposal.decision if self.proposal else None

    @property
    def cleared_to_build(self) -> bool:
        """The only state in which agent code may be written."""
        return (
            not self.problems
            and self.recommended == "build"
            and self.decision is not None
            and self.decision.outcome == "build"
        )

    @property
    def summary(self) -> str:
        if self.problems:
            return "proposal invalid: " + "; ".join(self.problems)
        written = self.decision.outcome if self.decision else "no decision written"
        return f"platform recommends {self.recommended}; decision: {written}"


# ── the rules ────────────────────────────────────────────────────────────────────────────


def max_autonomy(error: ErrorCost, rules: ReadinessRules) -> Tier:
    """How far the agent may go on its own, from what one error costs.

    act_alone           the error can be undone and costs no more than the platform limit
    act_after_approval  the error can be undone, or it is cheap
    advise              the error is expensive and cannot be undone: a person always acts
    """
    cheap = error.cost <= rules.act_alone_max_error_cost
    if error.reversible and cheap:
        return "act_alone"
    if error.reversible or cheap:
        return "act_after_approval"
    return "advise"


def tier_allows(cap: Tier, wanted: Tier) -> bool:
    return TIERS.index(wanted) <= TIERS.index(cap)


def load_reality_test(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"reality_test.jsonl line {n}: not JSON ({e.msg})") from e
            missing = {"case_id", "first", "second"} - row.keys()
            if missing:
                raise ValueError(f"reality_test.jsonl line {n}: missing {', '.join(missing)}")
            for who in ("first", "second"):
                if not isinstance(row[who], dict) or {"answer", "by_rules"} - row[who].keys():
                    raise ValueError(
                        f"reality_test.jsonl line {n}: {who} needs answer and by_rules"
                    )
            rows.append(row)
    ids = [r["case_id"] for r in rows]
    repeated = sorted({str(i) for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"reality_test.jsonl: repeated case ids: {', '.join(repeated)}")
    return rows


def score_reality_test(rows: list[dict]) -> RealityResult:
    def same(row: dict) -> bool:
        return canonical(row["first"]["answer"]) == canonical(row["second"]["answer"])

    def settled(row: dict) -> bool:
        return row["first"]["by_rules"] is True and row["second"]["by_rules"] is True

    n = len(rows) or 1
    return RealityResult(
        cases=len(rows),
        coverage=sum(map(settled, rows)) / n,
        agreement=sum(map(same, rows)) / n,
        disagreements=[str(r["case_id"]) for r in rows if not same(r)],
        unsettled=[str(r["case_id"]) for r in rows if not settled(r)],
    )


def _process_checks(p: Proposal, today: date) -> list[Check]:
    process = p.process
    gaps = []
    if len(process.steps) < 2:
        gaps.append("fewer than two steps")
    for step in process.steps:
        if not step.inputs:
            gaps.append(f"step {step.id} has no inputs")
        if not step.outputs:
            gaps.append(f"step {step.id} has no outputs")
    decision_ids = {d.id for d in process.decisions}
    used = {s.decision for s in process.steps if s.decision}
    if not process.decisions:
        gaps.append("no decisions written down")
    gaps += [f"step refers to unknown decision {d}" for d in sorted(used - decision_ids)]
    gaps += [f"decision {d} is not taken in any step" for d in sorted(decision_ids - used)]
    if not process.exceptions:
        gaps.append("no exceptions listed (every real process has some)")
    if not process.systems:
        gaps.append("no systems named")

    sign = process.signed_off
    if sign is None:
        signed, sign_detail = False, "not signed off"
    elif sign.by.casefold() != p.owner.name.casefold():
        signed, sign_detail = False, f"signed by {sign.by}, but the owner is {p.owner.name}"
    elif sign.signed_on > today:
        signed, sign_detail = False, f"sign-off dated in the future ({sign.signed_on})"
    else:
        signed, sign_detail = True, f"{sign.by}, {sign.signed_on}"

    return [
        Check(
            "process_map",
            "Process map is complete",
            not gaps,
            "; ".join(gaps)
            or f"{len(process.steps)} steps, {len(process.decisions)} decisions, "
            f"{len(process.exceptions)} exceptions, systems: {', '.join(process.systems)}",
        ),
        Check("sign_off", "Signed off by the process owner", signed, sign_detail),
    ]


def assess(agent_dir: Path, today: date | None = None) -> Readiness:
    today = today or datetime.now(UTC).date()
    rules = platform_config().readiness
    path = agent_dir / PROPOSAL_FILE
    if not path.is_file():
        return Readiness(agent_dir.name, None, problems=[f"no {PROPOSAL_FILE}"])
    try:
        proposal = Proposal.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except yaml.YAMLError as e:
        return Readiness(agent_dir.name, None, problems=[f"{PROPOSAL_FILE} is not YAML: {e}"])
    except ValidationError as e:
        problems = [
            f"{'.'.join(map(str, err['loc']))}: {err['msg'].removeprefix('Value error, ')}"
            for err in e.errors()
        ]
        return Readiness(agent_dir.name, None, problems=problems)

    result = Readiness(proposal.name, proposal)
    if proposal.name != agent_dir.name:
        result.problems.append(f"name {proposal.name!r} does not match the folder")

    checks = _process_checks(proposal, today)

    reality_path = agent_dir / REALITY_TEST_FILE
    reality: RealityResult | None = None
    if not reality_path.is_file():
        checks.append(Check("reality_test", "Reality test done", False, f"no {REALITY_TEST_FILE}"))
    else:
        try:
            reality = score_reality_test(load_reality_test(reality_path))
        except ValueError as e:
            checks.append(Check("reality_test", "Reality test done", False, str(e)))
    if reality is not None:
        staff = " and ".join(proposal.reality_test.staff)
        checks += [
            Check(
                "reality_test",
                "Reality test done",
                reality.cases >= rules.min_reality_cases,
                f"{reality.cases} past cases by {staff} (need {rules.min_reality_cases})",
            ),
            Check(
                "coverage",
                "Written rules settle enough cases",
                reality.coverage >= rules.min_coverage,
                f"{reality.coverage:.0%} settled by the rules (need {rules.min_coverage:.0%})"
                + (f"; not settled: {', '.join(reality.unsettled)}" if reality.unsettled else ""),
            ),
            Check(
                "agreement",
                "Two staff reach the same answer",
                reality.agreement >= rules.min_agreement,
                f"{reality.agreement:.0%} agreement (need {rules.min_agreement:.0%})"
                + (
                    f"; differed on: {', '.join(reality.disagreements)}"
                    if reality.disagreements
                    else ""
                ),
            ),
        ]

    cap = max_autonomy(proposal.error, rules)
    undo = "can be undone" if proposal.error.reversible else "cannot be undone"
    checks.append(
        Check(
            "autonomy",
            "Requested autonomy fits the cost of an error",
            tier_allows(cap, proposal.autonomy),
            f"asks for {proposal.autonomy}; an error costs {proposal.error.cost:g} and {undo}, "
            f"so the most allowed is {cap}",
        )
    )
    m, g = proposal.metric, proposal.guardrail
    checks.append(
        Check(
            "metric",
            "One metric with baseline and target",
            True,
            f"{m.name}: {m.baseline:g} -> {m.target:g} {m.unit}",
        )
    )
    checks.append(
        Check(
            "guardrail",
            "One guardrail metric",
            True,
            f"{g.name} must stay {g.must_stay} {g.limit:g} {g.unit} (now {g.current:g})",
        )
    )

    result.checks = checks
    result.reality = reality
    result.max_autonomy = cap
    result.recommended, result.reasons = _recommend(proposal, checks, reality, rules)
    decision_problem = _check_written_decision(proposal.decision, result.recommended, today)
    if decision_problem:
        result.problems.append(decision_problem)
    return result


def _recommend(
    p: Proposal, checks: list[Check], reality: RealityResult | None, rules: ReadinessRules
) -> tuple[Outcome, list[str]]:
    decline = []
    floor = break_even_accuracy(p.baseline.staff_cost_per_case, p.error.cost)
    if reality is not None and reality.cases and floor > reality.agreement:
        decline.append(
            f"break-even accuracy is {floor:.0%} but two staff agree on only "
            f"{reality.agreement:.0%}: no agent can be held to a bar people cannot meet"
        )
    coverage = reality.coverage if reality is not None else 1.0
    best = p.baseline.cases_per_month * coverage * p.baseline.staff_cost_per_case
    if best < rules.min_monthly_value:
        decline.append(
            f"even settling every rule-covered case, the saving is {best:,.0f} a month, "
            f"below the {rules.min_monthly_value:,.0f} worth building for"
        )
    if decline:
        return "decline", decline
    failed = [f"{c.label}: {c.detail}" for c in checks if not c.passed]
    if failed:
        return "fix_process", failed
    return "build", []


def _check_written_decision(
    decision: WrittenDecision | None, recommended: Outcome, today: date
) -> str | None:
    if decision is None:
        return None
    if decision.decided_on > today:
        return f"the decision is dated in the future ({decision.decided_on})"
    if decision.outcome == "build" and recommended != "build":
        return f"the decision says build, but the checks say {recommended}"
    return None


def passport_conflicts(r: Readiness, passport: Passport) -> list[str]:
    """Where the passport disagrees with the proposal it was cleared under."""
    p = r.proposal
    if p is None:
        return []
    conflicts = []
    if passport.baseline != p.baseline:
        conflicts.append("passport baseline differs from the proposal")
    if passport.error_cost != p.error.cost:
        conflicts.append(
            f"passport error_cost {passport.error_cost:g} differs from the proposal's "
            f"{p.error.cost:g}"
        )
    if passport.metric != p.metric:
        conflicts.append("passport metric differs from the proposal")
    if r.max_autonomy and not tier_allows(r.max_autonomy, passport.autonomy):
        conflicts.append(
            f"passport autonomy {passport.autonomy} is above the {r.max_autonomy} "
            "the readiness check allows"
        )
    if r.reality is not None and passport.exam.min_accuracy > r.reality.agreement:
        conflicts.append(
            f"exam.min_accuracy {passport.exam.min_accuracy:.0%} is above the staff agreement "
            f"of {r.reality.agreement:.0%}, which is its ceiling"
        )
    return conflicts


def has_agent_code(agent_dir: Path) -> bool:
    return any(p for p in agent_dir.glob("*.py"))


def as_dict(r: Readiness) -> dict[str, Any]:
    return {
        "name": r.name,
        "problems": r.problems,
        "checks": [c.__dict__ for c in r.checks],
        "coverage": r.reality.coverage if r.reality else None,
        "agreement": r.reality.agreement if r.reality else None,
        "max_autonomy": r.max_autonomy,
        "recommended": r.recommended,
        "reasons": r.reasons,
        "decision": r.decision.model_dump(mode="json") if r.decision else None,
        "cleared_to_build": r.cleared_to_build,
    }
