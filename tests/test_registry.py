import pytest

from agent_platform.exam import acceptance_report_path, run_exam, save_report
from agent_platform.registry import check_agent, find, record, records

from .conftest import make_agent


def accept(agent_dir, **kw):
    report = run_exam(agent_dir, **kw)
    save_report(report, acceptance_report_path(report.agent))
    return report


def test_fixture_folder_is_valid(workspace):
    assert check_agent(workspace / "agents" / "expense-coder") == []


def test_folder_problems_are_listed(workspace):
    agent = make_agent(workspace, "broken", model="gpt-9")
    (agent / "graph.py").write_text("x = 1\n", encoding="utf-8")
    problems = check_agent(agent)
    assert any("gpt-9" in p for p in problems)
    assert "graph.py has no build_graph()" in problems


def test_name_must_match_folder(workspace):
    agent = make_agent(workspace, "renamed")
    text = (agent / "agent.yaml").read_text(encoding="utf-8")
    (agent / "agent.yaml").write_text(text.replace("name: renamed", "name: other"), "utf-8")
    assert any("does not match the folder" in p for p in check_agent(agent))


@pytest.mark.parametrize(("lifecycle", "status"), [("draft", "draft"), ("retired", "retired")])
def test_lifecycle_statuses(workspace, lifecycle, status):
    agent = make_agent(workspace, "life", lifecycle=lifecycle)
    assert record(agent).status == status


def test_no_exam(workspace):
    assert record(workspace / "agents" / "expense-coder").status == "no_exam"


def test_mock_exam_is_never_accepted(workspace):
    agent = make_agent(workspace, "mocked", exam={"min_accuracy": 0.72, "min_holdout_cases": 10})
    accept(agent, model="mock")
    r = record(agent)
    assert r.status == "not_accepted"
    assert "mock" in r.reasons[0]


def test_running_after_a_passed_exam_on_the_passport_model(workspace, real_model):
    agent = make_agent(workspace, "expense-real", model="test-real")
    accept(agent)
    r = record(agent)
    assert r.status == "running", r.reasons
    assert r.runs


def test_provider_change_requires_another_exam(workspace, real_model):
    agent = make_agent(workspace, "expense-real", model="test-real")
    accept(agent)
    config = workspace / "config" / "models.yaml"
    config.write_text(config.read_text().replace("vendor/test-real-1", "vendor/test-real-2"))
    assert record(agent).status == "not_accepted"


def test_report_installer_rejects_mock_and_accepts_real(workspace, real_model):
    from agent_platform.cli import main

    agent = make_agent(workspace, "expense-real", model="test-real")
    path = workspace / "candidate.json"
    save_report(run_exam(agent, model="mock"), path)
    assert main(["accept-report", str(path)]) == 1
    save_report(run_exam(agent), path)
    assert main(["accept-report", str(path)]) == 0
    assert record(agent).runs


def test_exam_on_another_model_is_not_accepted(workspace, real_model):
    agent = make_agent(workspace, "expense-real", model="test-real")
    accept(agent, model="test-other")
    r = record(agent)
    assert r.status == "not_accepted"
    assert "ran on test-other" in r.reasons[0]


def test_changed_agent_needs_a_new_exam(workspace, real_model):
    agent = make_agent(workspace, "expense-real", model="test-real")
    accept(agent)
    graph = agent / "graph.py"
    graph.write_text(graph.read_text(encoding="utf-8").replace("LIMIT = 1000", "LIMIT = 5000"))
    r = record(agent)
    assert r.status == "not_accepted"
    assert "changed after its exam" in r.reasons[0]


def test_failed_exam_reasons_carry_over(workspace, real_model):
    agent = make_agent(
        workspace, "strict", model="test-real", exam={"min_accuracy": 0.9, "min_holdout_cases": 10}
    )
    accept(agent)
    r = record(agent)
    assert r.status == "not_accepted"
    assert "below the required 90.0%" in r.reasons[0]


def test_invalid_passport_is_not_accepted(workspace):
    agent = make_agent(workspace, "bad")
    (agent / "agent.yaml").write_text("name: bad\n", encoding="utf-8")
    r = record(agent)
    assert r.status == "not_accepted"
    assert r.passport is None


def test_unreadable_report(workspace):
    path = acceptance_report_path("expense-coder")
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    r = record(workspace / "agents" / "expense-coder")
    assert r.status == "not_accepted"
    assert "unreadable" in r.reasons[0]


def test_find_rejects_paths(workspace):
    assert find("expense-coder") is not None
    assert find("../agents/expense-coder") is None
    assert find("nobody") is None


def test_records_skip_hidden_and_template_folders(workspace):
    make_agent(workspace, "_template")
    assert [r.name for r in records()] == ["expense-coder"]
