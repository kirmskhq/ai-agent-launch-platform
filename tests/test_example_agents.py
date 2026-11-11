"""The two example agents: their folders pass every gate, their cases agree with their
written process, and their jobs read the sandbox."""

import json

import pytest

from agent_platform.exam import run_exam
from agent_platform.gateway import BadModelReply, mock_model, use_model
from agent_platform.loader import load_graph, load_job, load_mock_model
from agent_platform.readiness import assess
from agent_platform.registry import check_agent, record
from agent_platform.settings import ROOT
from agent_platform.tools import granted

AGENTS = ROOT / "agents"
RECON = AGENTS / "payment-reconciliation"
REFUNDS = AGENTS / "refund-requests"


@pytest.fixture(autouse=True)
def repo_agents(monkeypatch, workspace):
    monkeypatch.setenv("AGENTS_DIR", str(AGENTS))
    monkeypatch.setenv("AGENTS_CONFIG_DIR", str(ROOT / "config"))


def cases(agent, case_set):
    path = agent / "evals" / f"{case_set}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("agent", [RECON, REFUNDS], ids=lambda p: p.name)
def test_example_agent_passes_every_gate_before_the_exam(agent):
    assert check_agent(agent) == []
    readiness = assess(agent)
    assert readiness.cleared_to_build
    assert record(agent).status in ("no_exam", "running", "not_accepted")


def run(graph, case, reply: str):
    with use_model("mock"), mock_model(lambda messages: reply):
        return graph.invoke(dict(case["input"]))["output"]


@pytest.mark.parametrize("case_set", ["dev", "holdout"])
def test_reconciliation_cases_follow_the_written_process(case_set):
    """A model that proposes the right invoices must always get through the checks, and
    every handover case must be stopped by the checks whatever the model says."""
    graph = load_graph(RECON)
    for case in cases(RECON, case_set):
        expected = case["expected"]
        right = [] if expected.get("handover") else expected["invoice_ids"]
        out = run(graph, case, json.dumps({"invoice_ids": right}))
        if expected.get("handover"):
            assert out.get("handover"), case["id"]
            every_open = [i["id"] for i in case["input"]["open_invoices"]]
            assert run(graph, case, json.dumps({"invoice_ids": every_open})).get("handover")
        else:
            assert out["invoice_ids"] == expected["invoice_ids"], case["id"]
            assert out["fee"] == pytest.approx(expected["fee"]), case["id"]


@pytest.mark.parametrize("case_set", ["dev", "holdout"])
def test_refund_cases_follow_the_written_policy(case_set):
    graph = load_graph(REFUNDS)
    for case in cases(REFUNDS, case_set):
        out = run(graph, case, json.dumps({"reason": case["reason_in_message"]}))
        expected = case["expected"]
        if expected.get("handover"):
            assert out.get("handover"), case["id"]
        else:
            assert {k: out.get(k) for k in expected} == expected, case["id"]


def test_reconciliation_checks_stop_a_model_that_guesses():
    graph = load_graph(RECON)
    case = {
        "input": {
            "line": {
                "id": "x",
                "booked_on": "2026-10-01",
                "amount": 500.0,
                "payer": "p",
                "reference": "inv 1001",
            },
            "open_invoices": [
                {"id": "INV-1001", "amount": 500.0, "issued_on": "", "due_on": ""},
                {"id": "INV-1002", "amount": 500.0, "issued_on": "", "due_on": ""},
            ],
        }
    }
    out = run(graph, case, json.dumps({"invoice_ids": ["INV-1002"]}))
    assert out["handover"] and "does not name INV-1002" in out["reason"]
    with pytest.raises(BadModelReply):  # the runner turns this into a handover in production
        run(graph, case, "not json at all")


def test_mock_exams_never_accept(workspace):
    for agent in (RECON, REFUNDS):
        report = run_exam(agent, model="mock", runs_per_case=1)
        assert report.measured_on == "mock"
        assert report.counts["answered_wrong"] == 0
        assert report.counts["unneeded_handover"] > 0  # the keyword stand-in misreads some


def test_reconciliation_job_reads_the_sandbox():
    job = load_job(RECON)
    with granted(
        "payment-reconciliation",
        ["list_unmatched_statement_lines", "find_customer", "list_open_invoices"],
        can_write=False,
    ):
        inputs = job.collect()
    assert len(inputs) == 8
    by_line = {i["line"]["id"]: i for i in inputs}
    assert by_line["BSL-5007"]["open_invoices"] == []
    assert [i["id"] for i in by_line["BSL-5002"]["open_invoices"]] == ["INV-1201", "INV-1202"]

    graph = load_graph(RECON)
    with use_model("mock"), mock_model(load_mock_model(RECON)):
        out = {k: graph.invoke(dict(v))["output"] for k, v in by_line.items()}
    assert out["BSL-5001"] == {"invoice_ids": ["INV-1203"], "fee": 0.0, "method": "rules"}
    assert out["BSL-5002"]["invoice_ids"] == ["INV-1201", "INV-1202"]
    assert out["BSL-5002"]["fee"] == 12.5
    assert out["BSL-5004"]["fee"] == 15.0
    for handed_over in ("BSL-5003", "BSL-5005", "BSL-5007"):
        assert out[handed_over]["handover"], handed_over


def test_refund_job_reads_the_sandbox():
    job = load_job(REFUNDS)
    with granted("refund-requests", ["list_pending_refund_requests", "get_order"], can_write=False):
        inputs = job.collect()
    assert len(inputs) == 6
    assert {i["order"]["id"] for i in inputs} >= {"ORD-7001", "ORD-7003"}


def test_job_cannot_write_while_collecting():
    from agent_platform.tools import ToolRefused, call

    with granted("refund-requests", ["issue_refund"], can_write=False), pytest.raises(ToolRefused):
        call("issue_refund", request_id="RR-9001", amount=1)
