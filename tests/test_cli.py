import json

import yaml

from agent_platform import litellm_config
from agent_platform.cli import main
from agent_platform.compare import disagreements, render
from agent_platform.exam import load_report, run_exam, save_report
from agent_platform.settings import ROOT

from .conftest import make_agent


def test_validate(capsys):
    assert main(["validate"]) == 0
    assert "ok   expense-coder" in capsys.readouterr().out


def test_exam_writes_dev_reports_outside_the_acceptance_folder(workspace, capsys):
    assert main(["exam", "expense-coder", "--set", "dev", "--model", "mock"]) == 0
    out = capsys.readouterr().out
    assert "DEV expense-coder [mock, mock, dev 5 cases x 1 runs]" in out
    assert (workspace / "var" / "exams" / "expense-coder.dev.json").is_file()
    assert not (workspace / "reports" / "expense-coder.json").exists()


def test_failed_holdout_exam_exits_non_zero(workspace, capsys):
    assert main(["exam", "expense-coder", "--model", "mock"]) == 1
    out = capsys.readouterr().out
    assert "reason: answer accuracy 88.9%" in out
    assert "h11 [answered_wrong]" in out


def test_skip_without_key(workspace, capsys):
    make_agent(workspace, "expense-real", model="test-real")
    assert main(["exam", "expense-real", "--skip-without-key"]) == 0
    assert "skip expense-real: no key for test-real" in capsys.readouterr().out


def test_reuse_accepted_exam_requires_matching_fingerprint(
    workspace, real_model, monkeypatch, capsys
):
    from agent_platform import exam
    from agent_platform.exam import acceptance_report_path

    agent = make_agent(workspace, "expense-real", model="test-real")
    report = run_exam(agent)
    save_report(report, acceptance_report_path(report.agent))

    def unexpected_run(*args, **kwargs):
        raise RuntimeError("a fresh model run was requested")

    monkeypatch.setattr(exam, "run_exam", unexpected_run)
    assert main(["exam", "expense-real", "--reuse-accepted"]) == 0
    assert "reusing accepted exam" in capsys.readouterr().out
    graph = agent / "graph.py"
    graph.write_text(graph.read_text() + "\n# A new code version\n")
    import pytest

    with pytest.raises(RuntimeError, match="fresh model run"):
        main(["exam", "expense-real", "--reuse-accepted"])


def test_compare_two_models(workspace, real_model, capsys):
    agent = make_agent(workspace, "expense-real", model="test-real")
    a = run_exam(agent, model="mock")
    b = run_exam(agent)
    save_report(a, workspace / "a.json")
    save_report(b, workspace / "b.json")
    a, b = load_report(workspace / "a.json"), load_report(workspace / "b.json")
    assert ("h11", "answered_wrong", "answered_correct") in disagreements(a, b)
    text = render(a, b)
    assert "mock" in text and "test-real" in text
    assert main(["compare", str(workspace / "a.json"), str(workspace / "b.json")]) == 0
    assert "unneeded handover" in capsys.readouterr().out


def test_status(capsys):
    assert main(["status"]) == 0
    assert "no exam" in capsys.readouterr().out


def test_schema(capsys):
    assert main(["schema"]) == 0
    assert "baseline" in json.loads(capsys.readouterr().out)["properties"]


def test_litellm_config_leaves_out_the_mock(workspace):
    doc = yaml.safe_load(litellm_config.build())
    names = [m["model_name"] for m in doc["model_list"]]
    assert sorted(names) == ["test-other", "test-real"]
    keys = {m["litellm_params"]["api_key"] for m in doc["model_list"]}
    assert keys == {"os.environ/TESTCLOUD_KEY"}


def test_committed_litellm_config_is_current(monkeypatch):
    monkeypatch.setenv("AGENTS_CONFIG_DIR", str(ROOT / "config"))
    assert litellm_config.is_current(), "run `uv run agents litellm-config`"
