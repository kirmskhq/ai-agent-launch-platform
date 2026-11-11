"""The agent passport (agent.yaml): what a department signs up for before anything runs.

A passport answers the questions an acceptance review asks: whose agent it is, what work it
replaces, what that work costs today, what one error costs, which metric it should move,
which model it uses and what bar it has to clear on the exam.

The passport is strict: unknown fields are errors, not silently ignored, and an accuracy
threshold below break-even is rejected outright, because at that threshold the agent would be
accepted while losing money.
"""

from pathlib import Path
from typing import Literal

import yaml
from croniter import croniter
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent_platform.economics import break_even_accuracy, staff_cost_per_case

NAME_PATTERN = r"^[a-z][a-z0-9-]{1,40}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Baseline(_Strict):
    """How the work is done today, by people."""

    cases_per_month: int = Field(gt=0)
    minutes_per_case: float = Field(gt=0, description="Staff minutes to settle one case")
    staff_cost_per_hour: float = Field(gt=0, description="Fully loaded, in platform currency")

    @property
    def staff_cost_per_case(self) -> float:
        return staff_cost_per_case(self.minutes_per_case, self.staff_cost_per_hour)


class Metric(_Strict):
    """The one business number the agent is meant to move."""

    name: str = Field(min_length=3)
    unit: str = Field(min_length=1)
    baseline: float
    target: float

    @model_validator(mode="after")
    def _target_differs(self) -> "Metric":
        if self.target == self.baseline:
            raise ValueError("target equals baseline: the agent would change nothing")
        return self

    @property
    def lower_is_better(self) -> bool:
        return self.target < self.baseline


class Exam(_Strict):
    """How the agent is accepted on the holdout cases."""

    scoring: Literal["fields", "exact"] = Field(
        default="fields",
        description="fields: the listed fields (default: every field of the expected answer) "
        "must match; exact: the whole answer must match",
    )
    fields: list[str] | None = None
    min_accuracy: float = Field(
        ge=0,
        le=1,
        description="The lower confidence bound of answer accuracy must reach this",
    )
    confidence: float = Field(default=0.90, gt=0.5, lt=1)
    runs_per_case: int = Field(default=3, ge=1, le=10)
    min_holdout_cases: int = Field(default=30, ge=10)
    min_answered_share: float = Field(
        default=0.0, ge=0, le=1, description="Share of cases the agent must settle itself"
    )
    max_cost_per_case: float | None = Field(
        default=None, gt=0, description="Ceiling on model cost per case, platform currency"
    )

    @model_validator(mode="after")
    def _fields_not_empty(self) -> "Exam":
        if self.fields == []:
            raise ValueError("exam.fields is empty; remove it to compare every expected field")
        if self.scoring == "exact" and self.fields:
            raise ValueError("exam.fields only applies to scoring: fields")
        return self


class Passport(_Strict):
    name: str = Field(pattern=NAME_PATTERN)
    title: str = Field(min_length=3)
    department: str = Field(min_length=2)
    owner: str = Field(min_length=3, description="Process owner who answers for the agent")
    replaces: str = Field(min_length=10, description="The work the agent takes over, plainly")
    baseline: Baseline
    error_cost: float = Field(ge=0, description="Cost of one wrong answer that goes out")
    metric: Metric
    model: str = Field(description="Alias from config/models.yaml")
    autonomy: Literal["advise", "act_after_approval", "act_alone"] = Field(
        default="advise",
        description="advise: a person acts on the answer; act_after_approval: the agent acts "
        "once a person approves; act_alone: the agent acts. Capped by the readiness check.",
    )
    schedule: str | None = Field(default=None, description="cron, 5 fields, UTC")
    lifecycle: Literal["draft", "live", "retired"] = Field(
        default="draft",
        description="draft: being built; live: should run once accepted; retired: switched off",
    )
    exam: Exam

    @model_validator(mode="after")
    def _check(self) -> "Passport":
        floor = self.break_even_accuracy
        if self.exam.min_accuracy < floor:
            raise ValueError(
                f"exam.min_accuracy {self.exam.min_accuracy:.1%} is below break-even "
                f"{floor:.1%}: with an error costing {self.error_cost:g} and a case worth "
                f"{self.baseline.staff_cost_per_case:.2f} in staff time, an agent at this "
                "threshold loses money on every case it answers"
            )
        if self.schedule is not None and (
            len(self.schedule.split()) != 5 or not croniter.is_valid(self.schedule)
        ):
            raise ValueError(f"schedule {self.schedule!r} is not a 5-field cron expression")
        return self

    @property
    def break_even_accuracy(self) -> float:
        return break_even_accuracy(self.baseline.staff_cost_per_case, self.error_cost)


class PassportError(ValueError):
    """agent.yaml is missing, is not YAML or does not satisfy the passport rules."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def load_passport(agent_dir: Path) -> Passport:
    path = agent_dir / "agent.yaml"
    if not path.is_file():
        raise PassportError(["no agent.yaml"])
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise PassportError([f"agent.yaml is not valid YAML: {e}"]) from e
    try:
        return Passport.model_validate(data)
    except ValidationError as e:
        raise PassportError([_describe(err) for err in e.errors()]) from e


def _describe(err: dict) -> str:
    where = ".".join(str(p) for p in err["loc"])
    message = err["msg"].removeprefix("Value error, ")
    return f"{where}: {message}" if where else message


def json_schema() -> dict:
    """Passport JSON Schema, for editor hints."""
    return Passport.model_json_schema()
