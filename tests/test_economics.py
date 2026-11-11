import pytest

from agent_platform.economics import break_even_accuracy, monthly_effect
from agent_platform.stats import wilson_lower_bound


def effect(**kw):
    args = dict(
        cases_per_month=1000,
        minutes_per_case=6,
        staff_cost_per_hour=50,
        error_cost=20,
        correct_share=0.0,
        wrong_share=0.0,
        model_cost_per_case=0.0,
        hours_per_staff_month=160,
    )
    return monthly_effect(**(args | kw))


def test_break_even():
    # a right answer saves 5, a wrong one costs 20: answering pays from 80% accuracy
    assert break_even_accuracy(5, 20) == pytest.approx(0.8)
    assert break_even_accuracy(5, 0) == 0
    assert break_even_accuracy(0, 0) == 0


def test_handovers_save_nothing():
    e = effect(correct_share=0.0, wrong_share=0.0)
    assert e.handed_over == 1000
    assert e.staff_saving == 0
    assert e.net == 0


def test_effect_adds_up():
    e = effect(correct_share=0.7, wrong_share=0.1, model_cost_per_case=0.01)
    assert e.answered_correct == pytest.approx(700)
    assert e.answered_wrong == pytest.approx(100)
    assert e.handed_over == pytest.approx(200)
    assert e.staff_hours_saved == pytest.approx(70)
    assert e.staff_saving == pytest.approx(3500)
    assert e.error_cost == pytest.approx(2000)
    assert e.model_cost == pytest.approx(10)  # paid on every case, handed over or not
    assert e.net == pytest.approx(1490)
    assert e.staff_months_saved == pytest.approx(70 / 160)


def test_break_even_accuracy_gives_zero_net():
    a = break_even_accuracy(5, 20)
    assert effect(correct_share=a, wrong_share=1 - a).net == pytest.approx(0, abs=1e-9)


def test_shares_must_fit():
    with pytest.raises(ValueError):
        effect(correct_share=0.8, wrong_share=0.3)


def test_wilson_lower_bound_known_values():
    assert wilson_lower_bound(19, 20, 0.90) == pytest.approx(0.8468, abs=1e-3)
    assert wilson_lower_bound(10, 10, 0.90) == pytest.approx(0.8589, abs=1e-3)
    assert wilson_lower_bound(0, 10, 0.90) == 0


def test_wilson_lower_bound_tightens_with_more_cases():
    small = wilson_lower_bound(9, 10)
    large = wilson_lower_bound(900, 1000)
    assert small < large < 0.9


def test_wilson_lower_bound_edges():
    assert wilson_lower_bound(0, 0) == 0
    with pytest.raises(ValueError):
        wilson_lower_bound(11, 10)
    with pytest.raises(ValueError):
        wilson_lower_bound(5, 10, confidence=0.3)
