import shutil

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


@pytest.mark.parametrize("dependency", ["sandbox/refund_policy.md", "pyproject.toml", "uv.lock"])
def test_runtime_dependency_change_invalidates_acceptance(
    workspace, real_model, monkeypatch, dependency
):
    from agent_platform import settings

    # Isolate shared runtime files so the test never edits the checkout.
    runtime = workspace / "runtime"
    for folder in ("agent_platform", "sandbox"):
        shutil.copytree(settings.ROOT / folder, runtime / folder)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy(settings.ROOT / name, runtime / name)
    monkeypatch.setattr(settings, "ROOT", runtime)

    agent = make_agent(workspace, "expense-real", model="test-real")
    accept(agent)
    assert record(agent).runs

    path = runtime / dependency
    original = path.read_bytes()
    # Line-ending conversion alone must preserve cross-platform acceptance.
    path.write_bytes(original.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert record(agent).runs

    path.write_bytes(original + b"\n# Changed runtime dependency\n")
    changed = record(agent)
    assert changed.status == "not_accepted"
    assert "changed after its exam" in changed.reasons[0]

    # Restoring the exact dependency restores the original acceptance identity.
    path.write_bytes(original)
    assert record(agent).runs


def _settle(*folders):
    """Date every file a minute back, as if it was written long before the request."""
    import os
    import time

    past = time.time() - 60
    for folder in folders:
        for path in folder.rglob("*"):
            if path.is_file():
                os.utime(path, (past, past))


def test_record_is_reused_until_a_file_changes(workspace, real_model):
    agent = make_agent(workspace, "expense-real", model="test-real")
    accept(agent)
    _settle(workspace)
    first = record(agent)
    assert first.runs
    assert record(agent) is first
    graph = agent / "graph.py"
    graph.write_text(graph.read_text(encoding="utf-8").replace("LIMIT = 1000", "LIMIT = 5000"))
    assert record(agent).status == "not_accepted"


def test_fresh_files_are_never_trusted_to_their_timestamp(workspace):
    agent = workspace / "agents" / "expense-coder"
    assert record(agent) is not record(agent)


def test_report_removal_and_a_new_day_refresh_the_record(workspace, real_model, monkeypatch):
    from agent_platform import registry

    agent = make_agent(workspace, "expense-real", model="test-real")
    accept(agent)
    _settle(workspace)
    first = record(agent)
    monkeypatch.setattr(registry, "_today", lambda: "2999-01-01")
    assert record(agent) is not first
    acceptance_report_path("expense-real").unlink()
    assert record(agent).status == "no_exam"
