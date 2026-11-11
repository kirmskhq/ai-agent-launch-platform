"""Two exam reports side by side, for example the same agent on two models."""

from dataclasses import dataclass

from agent_platform.exam import ExamReport


@dataclass(frozen=True)
class Row:
    label: str
    a: str
    b: str


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def _money(x: float, currency: str, digits: int = 0) -> str:
    return f"{x:,.{digits}f} {currency}"


def rows(a: ExamReport, b: ExamReport) -> list[Row]:
    def row(label: str, fn) -> Row:
        return Row(label, fn(a), fn(b))

    return [
        row("model", lambda r: r.model),
        row("measured on", lambda r: r.measured_on),
        row("case set", lambda r: f"{r.case_set}, {r.cases_total} cases x {r.runs_per_case}"),
        row("passed", lambda r: "yes" if r.passed else "no"),
        row("answered correct", lambda r: str(r.counts["answered_correct"])),
        row("answered wrong", lambda r: str(r.counts["answered_wrong"])),
        row("correct handover", lambda r: str(r.counts["correct_handover"])),
        row("unneeded handover", lambda r: str(r.counts["unneeded_handover"])),
        row("infrastructure failure", lambda r: str(r.counts["infra_failure"])),
        row("answer accuracy", lambda r: _pct(r.answer_accuracy)),
        row(
            "accuracy lower bound",
            lambda r: f"{_pct(r.accuracy_lower_bound)} (need {_pct(r.min_accuracy)})",
        ),
        row("settled by the agent", lambda r: _pct(r.answered_share)),
        row("consistency", lambda r: _pct(r.consistency)),
        row("model cost per case", lambda r: _money(r.cost_per_case, r.currency, 4)),
        row("latency p50 / p95", lambda r: f"{r.latency_p50_s:.1f}s / {r.latency_p95_s:.1f}s"),
        row("monthly effect", lambda r: _money(r.effect["net"], r.currency)),
        row("at lower bound", lambda r: _money(r.effect_at_lower_bound["net"], r.currency)),
    ]


def disagreements(a: ExamReport, b: ExamReport) -> list[tuple[str, str, str]]:
    """Cases present in both reports whose outcome differs: (case id, outcome a, outcome b)."""
    b_cases = {c.id: c for c in b.cases}
    return [
        (c.id, c.outcome, b_cases[c.id].outcome)
        for c in a.cases
        if c.id in b_cases and c.outcome != b_cases[c.id].outcome
    ]


def render(a: ExamReport, b: ExamReport) -> str:
    table = rows(a, b)
    w0 = max(len(r.label) for r in table)
    w1 = max(len(r.a) for r in table)
    lines = [f"{r.label:<{w0}}  {r.a:<{w1}}  {r.b}" for r in table]
    if a.agent != b.agent:
        lines.insert(0, f"note: different agents ({a.agent} vs {b.agent})")
    if a.case_set != b.case_set:
        lines.insert(0, "note: different case sets, the numbers are not comparable")
    diff = disagreements(a, b)
    if diff:
        lines.append("")
        lines.append(f"cases that ended differently ({len(diff)}):")
        lines.extend(f"  {case_id}: {oa} -> {ob}" for case_id, oa, ob in diff)
    return "\n".join(lines)
