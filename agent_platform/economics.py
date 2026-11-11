"""What an agent is worth per month, in the platform currency.

The rules are deliberately plain:

- A case the agent answers correctly saves the staff time the baseline says it takes.
- A case the agent answers wrongly costs `error_cost` (the passport's cost of one error that
  goes out unnoticed). We assume nobody catches it; that is the cautious assumption.
- A case the agent hands to a person saves nothing: the person still does the work. This holds
  even when the handover was the right call.
- The model is paid for on every case, including the ones handed over.
"""

from dataclasses import asdict, dataclass


def staff_cost_per_case(minutes_per_case: float, staff_cost_per_hour: float) -> float:
    return minutes_per_case / 60 * staff_cost_per_hour


def break_even_accuracy(saving_per_case: float, error_cost: float) -> float:
    """Accuracy below which answering instead of handing over loses money.

    A right answer saves s, a wrong one costs e. Answering pays when a*s - (1 - a)*e >= 0,
    that is when a >= e / (s + e). Model cost is left out here because it is paid on handed-over
    cases too; the exam adds it to the monthly effect.
    """
    if saving_per_case + error_cost <= 0:
        return 0.0
    return error_cost / (saving_per_case + error_cost)


@dataclass(frozen=True)
class MonthlyEffect:
    cases: int
    answered_correct: float
    answered_wrong: float
    handed_over: float
    staff_hours_saved: float
    staff_months_saved: float
    staff_saving: float
    error_cost: float
    model_cost: float
    net: float

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def monthly_effect(
    *,
    cases_per_month: int,
    minutes_per_case: float,
    staff_cost_per_hour: float,
    error_cost: float,
    correct_share: float,
    wrong_share: float,
    model_cost_per_case: float,
    hours_per_staff_month: float,
) -> MonthlyEffect:
    """`correct_share` and `wrong_share` are shares of all cases, answered by the agent."""
    if not 0 <= correct_share + wrong_share <= 1 + 1e-9:
        raise ValueError("correct_share + wrong_share must be between 0 and 1")
    correct = cases_per_month * correct_share
    wrong = cases_per_month * wrong_share
    hours = correct * minutes_per_case / 60
    saving = hours * staff_cost_per_hour
    errors = wrong * error_cost
    model = cases_per_month * model_cost_per_case
    return MonthlyEffect(
        cases=cases_per_month,
        answered_correct=correct,
        answered_wrong=wrong,
        handed_over=cases_per_month - correct - wrong,
        staff_hours_saved=hours,
        staff_months_saved=hours / hours_per_staff_month,
        staff_saving=saving,
        error_cost=errors,
        model_cost=model,
        net=saving - errors - model,
    )
