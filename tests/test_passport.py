import pytest
import yaml
from pydantic import ValidationError

from agent_platform.passport import Passport, PassportError, load_passport

from .conftest import FIXTURES


def base() -> dict:
    path = FIXTURES / "agents" / "expense-coder" / "agent.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_fixture_passport_is_valid():
    p = Passport.model_validate(base())
    assert p.baseline.staff_cost_per_case == pytest.approx(2.0)
    assert p.break_even_accuracy == pytest.approx(5 / 7)
    assert p.exam.runs_per_case == 3
    assert p.exam.confidence == 0.90


def test_unknown_field_is_an_error():
    data = base() | {"budget": 100}
    with pytest.raises(ValidationError, match="budget"):
        Passport.model_validate(data)


def test_unknown_nested_field_is_an_error():
    data = base()
    data["baseline"]["hours"] = 3
    with pytest.raises(ValidationError):
        Passport.model_validate(data)


def test_threshold_below_break_even_is_rejected():
    data = base()
    data["exam"]["min_accuracy"] = 0.70  # break-even is 71.4%
    with pytest.raises(ValidationError, match="below break-even 71.4%"):
        Passport.model_validate(data)


def test_threshold_at_break_even_is_allowed():
    data = base()
    data["error_cost"] = 2.0  # break-even 50%
    data["exam"]["min_accuracy"] = 0.5
    assert Passport.model_validate(data).exam.min_accuracy == 0.5


def test_zero_error_cost_means_zero_break_even():
    data = base()
    data["error_cost"] = 0
    data["exam"]["min_accuracy"] = 0
    assert Passport.model_validate(data).break_even_accuracy == 0


@pytest.mark.parametrize("schedule", ["every day", "0 7 * *", "61 * * * *"])
def test_bad_schedule_is_rejected(schedule):
    with pytest.raises(ValidationError, match="cron"):
        Passport.model_validate(base() | {"schedule": schedule})


def test_metric_target_must_differ_from_baseline():
    data = base()
    data["metric"]["target"] = data["metric"]["baseline"]
    with pytest.raises(ValidationError, match="target equals baseline"):
        Passport.model_validate(data)


def test_name_pattern():
    with pytest.raises(ValidationError):
        Passport.model_validate(base() | {"name": "Expense Coder"})


def test_fields_list_cannot_be_empty():
    data = base()
    data["exam"]["fields"] = []
    with pytest.raises(ValidationError, match="empty"):
        Passport.model_validate(data)


def test_load_passport_lists_every_problem(tmp_path):
    data = base()
    data["baseline"]["cases_per_month"] = 0
    data["surprise"] = True
    (tmp_path / "agent.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(PassportError) as err:
        load_passport(tmp_path)
    assert any("cases_per_month" in p for p in err.value.problems)
    assert any("surprise" in p for p in err.value.problems)


def test_load_passport_without_file(tmp_path):
    with pytest.raises(PassportError, match="no agent.yaml"):
        load_passport(tmp_path)
