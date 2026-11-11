import json
import shutil
from datetime import date

import pytest
import yaml
from fastapi.testclient import TestClient

from agent_platform.cli import main
from agent_platform.config import ReadinessRules
from agent_platform.readiness import ErrorCost, assess, max_autonomy
from agent_platform.registry import check_agent, proposals, record
from agent_platform.service import create_app

from .conftest import make_agent, set_proposal

TODAY = date(2026, 11, 11)


def checks(r) -> dict[str, bool]:
    return {c.id: c.passed for c in r.checks}


def write_reality(agent_dir, n=30, disagree=(), not_by_rules=()):
    rows = []
    for i in range(1, n + 1):
        answer = {"category": "travel"}
        other = {"category": "meals"} if i in disagree else answer
        rows.append(
            {
                "case_id": f"rt-{i:02d}",
                "first": {"answer": answer, "by_rules": i not in not_by_rules},
                "second": {"answer": other, "by_rules": True},
            }
        )
    path = agent_dir / "readiness" / "reality_test.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def proposal_only(workspace, name="proposal-only"):
    """A folder with a proposal and reality test but no passport or code."""
    src = workspace / "agents" / "expense-coder"
    dst = workspace / "agents" / name
    dst.mkdir()
    shutil.copy(src / "proposal.yaml", dst / "proposal.yaml")
    shutil.copytree(src / "readiness", dst / "readiness")
    set_proposal(dst, name=name)
    return dst


@pytest.fixture
def agent(workspace):
    return workspace / "agents" / "expense-coder"


def test_fixture_is_cleared_to_build(agent):
    r = assess(agent, TODAY)
    assert r.problems == []
    assert all(checks(r).values()), r.checks
    assert r.reality.cases == 30
    assert r.reality.coverage == pytest.approx(26 / 30)
    assert r.reality.agreement == pytest.approx(27 / 30)
    assert r.reality.disagreements == ["rt-07", "rt-19", "rt-28"]
    assert r.max_autonomy == "act_alone"
    assert r.recommended == "build"
    assert r.cleared_to_build


@pytest.mark.parametrize(
    ("cost", "reversible", "tier"),
    [
        (10, True, "act_alone"),
        (50, True, "act_alone"),
        (500, True, "act_after_approval"),
        (10, False, "act_after_approval"),
        (500, False, "advise"),
    ],
)
def test_max_autonomy(cost, reversible, tier):
    error = ErrorCost(cost=cost, reversible=reversible, how_undone="refund" if reversible else None)
    assert max_autonomy(error, ReadinessRules()) == tier


def edit_process(agent, fn) -> None:
    path = agent / "proposal.yaml"
    proposal = yaml.safe_load(path.read_text(encoding="utf-8"))
    fn(proposal["process"])
    proposal.pop("decision", None)
    path.write_text(yaml.safe_dump(proposal), encoding="utf-8")


def test_missing_sign_off_means_fix_process(agent):
    edit_process(agent, lambda process: process.pop("signed_off"))
    r = assess(agent, TODAY)
    assert checks(r)["sign_off"] is False
    assert r.recommended == "fix_process"
    assert not r.cleared_to_build


@pytest.mark.parametrize(
    ("sign_off", "detail"),
    [
        ({"by": "Priya Raman", "signed_on": "2026-09-14"}, "the owner is Dana Whitfield"),
        ({"by": "Dana Whitfield", "signed_on": "2027-01-01"}, "in the future"),
    ],
)
def test_sign_off_by_the_owner_and_not_in_the_future(agent, sign_off, detail):
    edit_process(agent, lambda process: process.update(signed_off=sign_off))
    check = next(c for c in assess(agent, TODAY).checks if c.id == "sign_off")
    assert not check.passed
    assert detail in check.detail


def test_incomplete_process_map(agent):
    def damage(process):
        process["exceptions"] = []
        process["steps"][0]["outputs"] = []
        process["decisions"].append(
            {"id": "unused", "question": "Is it odd?", "rule": "Odd lines go to the controller."}
        )

    edit_process(agent, damage)
    detail = next(c for c in assess(agent, TODAY).checks if c.id == "process_map").detail
    assert "no exceptions listed" in detail
    assert "step read has no outputs" in detail
    assert "decision unused is not taken in any step" in detail


def test_low_coverage_means_fix_process(agent):
    write_reality(agent, not_by_rules=range(1, 11))  # rules settle 20 of 30
    set_proposal(agent, decision=None)
    r = assess(agent, TODAY)
    assert not checks(r)["coverage"]
    assert r.recommended == "fix_process"
    assert any("67% settled by the rules" in reason for reason in r.reasons)


def test_too_few_reality_cases(agent):
    write_reality(agent, n=12)
    set_proposal(agent, decision=None)
    assert not checks(assess(agent, TODAY))["reality_test"]


def test_agreement_below_break_even_means_decline(agent):
    # break-even is 5 / (2 + 5) = 71%; staff agree on 60%
    write_reality(agent, disagree=range(1, 13))
    set_proposal(agent, decision=None)
    r = assess(agent, TODAY)
    assert r.recommended == "decline"
    assert any("agree on only 60%" in reason for reason in r.reasons)


def test_too_little_value_means_decline(agent):
    set_proposal(
        agent,
        baseline={"cases_per_month": 50, "minutes_per_case": 3, "staff_cost_per_hour": 40},
        decision=None,
    )
    r = assess(agent, TODAY)
    assert r.recommended == "decline"
    assert any("below the 1,000 worth building for" in reason for reason in r.reasons)


def test_autonomy_cannot_exceed_what_the_error_allows(agent):
    set_proposal(
        agent,
        error={"cost": 500, "reversible": False, "how_undone": None},
        decision=None,
    )
    r = assess(agent, TODAY)
    assert r.max_autonomy == "advise"
    assert not checks(r)["autonomy"]
    assert r.recommended in ("fix_process", "decline")


def test_written_build_needs_the_checks_to_agree(agent):
    write_reality(agent, not_by_rules=range(1, 11))
    r = assess(agent, TODAY)
    assert "the decision says build, but the checks say fix_process" in r.problems
    assert not r.cleared_to_build


def test_written_decline_may_be_stricter(agent):
    set_proposal(
        agent,
        decision={
            "outcome": "decline",
            "reason": "The team is moving to a new ledger next quarter.",
            "by": "Dana Whitfield",
            "decided_on": "2026-09-20",
        },
    )
    r = assess(agent, TODAY)
    assert r.recommended == "build"
    assert r.problems == []
    assert not r.cleared_to_build


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"error": {"cost": 5, "reversible": True, "how_undone": None}}, "how_undone"),
        (
            {
                "guardrail": {
                    "name": "Recoded lines",
                    "unit": "percent",
                    "current": 5,
                    "must_stay": "below",
                    "limit": 4,
                }
            },
            "already breached",
        ),
        ({"reality_test": {"staff": ["Priya Raman", "priya raman"]}}, "two different"),
        ({"budget": 10}, "budget"),
    ],
)
def test_invalid_proposals(agent, change, message):
    set_proposal(agent, **change)
    r = assess(agent, TODAY)
    assert r.proposal is None
    assert any(message in p for p in r.problems), r.problems


# ── the gate in the registry ─────────────────────────────────────────────────────────────


def test_code_before_readiness_is_an_error(workspace):
    agent = make_agent(workspace, "early", lifecycle="draft")
    set_proposal(agent, decision=None)
    assert any("code exists before readiness" in p for p in check_agent(agent))


def test_code_without_any_proposal_is_an_error(workspace):
    agent = make_agent(workspace, "no-proposal")
    (agent / "proposal.yaml").unlink()
    assert any("no proposal.yaml" in p for p in check_agent(agent))
    assert record(agent).status == "not_accepted"


def test_live_agent_needs_readiness(workspace, real_model):
    agent = make_agent(workspace, "live-early", model="test-real")
    set_proposal(agent, decision=None)
    r = record(agent)
    assert r.status == "not_accepted"
    assert r.reasons[0].startswith("readiness:")


@pytest.mark.parametrize(
    ("passport_change", "message"),
    [
        ({"exam": {"min_accuracy": 0.95, "min_holdout_cases": 10}}, "above the staff agreement"),
        ({"error_cost": 4}, "error_cost 4 differs"),
    ],
)
def test_passport_must_match_its_proposal(workspace, passport_change, message):
    agent = make_agent(workspace, "mismatch", **passport_change)
    assert any(message in p for p in check_agent(agent))
    assert record(agent).status == "not_accepted"


def test_passport_autonomy_is_capped(workspace):
    agent = make_agent(workspace, "bold", error_cost=60, exam={"min_accuracy": 0.97})
    set_proposal(
        agent,
        error={"cost": 60, "reversible": True, "how_undone": "Recode it."},
        autonomy="act_after_approval",
    )
    problems = check_agent(agent)
    assert any("autonomy act_alone is above the act_after_approval" in p for p in problems)


def test_proposals_are_listed_apart(workspace):
    proposal_only(workspace)
    assert [r.name for r in proposals()] == ["proposal-only"]


def test_propose_and_check(workspace, capsys):
    assert main(["propose", "invoice-matching"]) == 0
    folder = workspace / "agents" / "invoice-matching"
    assert (folder / "proposal.yaml").is_file()
    assert (folder / "readiness" / "reality_test.jsonl").is_file()
    assert main(["readiness", "invoice-matching"]) == 0
    out = capsys.readouterr().out
    assert "cleared to build: no" in out
    assert "FAIL  Signed off by the process owner" in out
    assert main(["propose", "invoice-matching"]) == 1


def test_validate_covers_proposals(workspace, capsys):
    proposal_only(workspace)
    assert main(["validate"]) == 0
    assert "ok   proposal-only (proposal)" in capsys.readouterr().out


def test_catalogue_shows_proposals_and_readiness(workspace):
    proposal_only(workspace)
    with TestClient(create_app(scheduler=False)) as client:
        index = client.get("/").text
        assert "Proposals at the readiness checkpoint" in index
        page = client.get("/proposals/proposal-only")
        assert page.status_code == 200
        assert "Agreement between two staff" in page.text
        assert "Amounts above 1,000 always go to the controller" in page.text
        assert client.get("/proposals/nobody").status_code == 404
        assert "Readiness checkpoint" in client.get("/agents/expense-coder").text
        api = client.get("/api/proposals").json()
        assert api[0]["recommended"] == "build"
        assert client.get("/api/agents/expense-coder").json()["readiness"]["cleared_to_build"]
