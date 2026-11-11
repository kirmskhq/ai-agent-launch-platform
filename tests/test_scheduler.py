import threading
from datetime import UTC, datetime, timedelta, timezone

import pytest

from agent_platform.exam import acceptance_report_path, run_exam, save_report
from agent_platform.scheduler import Scheduler, is_due

from .conftest import make_agent


def test_schedules_are_in_utc():
    # 09:00 in UTC+3 is 06:00 UTC
    local = datetime(2026, 3, 2, 9, 0, 30, tzinfo=timezone(timedelta(hours=3)))
    assert is_due("0 6 * * *", local)
    assert not is_due("0 9 * * *", local)


def test_naive_time_is_refused():
    with pytest.raises(ValueError):
        is_due("* * * * *", datetime(2026, 3, 2, 6, 0))


@pytest.fixture
def scheduled(workspace, real_model):
    agent = make_agent(workspace, "nightly", model="test-real", schedule="0 6 * * *")
    save_report(run_exam(agent), acceptance_report_path("nightly"))
    make_agent(workspace, "draft-nightly", schedule="0 6 * * *", lifecycle="draft")
    return agent


def test_fires_once_per_minute_and_only_running_agents(scheduled):
    done = threading.Event()
    calls = []

    def fake_run(name, trigger):
        calls.append((name, trigger))
        done.set()
        return {"cases": 0, "handed_over": 0}

    s = Scheduler(run=fake_run)
    at = datetime(2026, 3, 2, 6, 0, 5, tzinfo=UTC)
    assert s.tick(at) == ["nightly"]
    assert done.wait(2)
    assert s.tick(at + timedelta(seconds=20)) == []  # same minute
    assert s.tick(at + timedelta(minutes=1)) == []  # not due
    assert calls == [("nightly", "schedule")]


def test_does_not_overlap_a_run_still_going(scheduled):
    release = threading.Event()

    def slow_run(name, trigger):
        release.wait(2)
        return {"cases": 0, "handed_over": 0}

    s = Scheduler(run=slow_run)
    day1 = datetime(2026, 3, 2, 6, 0, tzinfo=UTC)
    assert s.tick(day1) == ["nightly"]
    assert s.is_running("nightly")
    assert s.tick(day1 + timedelta(days=1)) == []
    release.set()
