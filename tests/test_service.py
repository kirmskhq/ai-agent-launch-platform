import pytest
from fastapi.testclient import TestClient

from agent_platform.exam import acceptance_report_path, run_exam, save_report
from agent_platform.service import create_app

from .conftest import make_agent


@pytest.fixture
def client():
    with TestClient(create_app(scheduler=False)) as c:
        yield c


@pytest.fixture
def running_agent(workspace, real_model):
    agent = make_agent(workspace, "expense-real", model="test-real")
    report = run_exam(agent)
    save_report(report, acceptance_report_path(report.agent))
    return agent.name


def auth(token="secret"):
    return {"Authorization": f"Bearer {token}"}


def test_staff_reviews_and_value_are_scoped_to_the_token(client, monkeypatch, running_agent):
    monkeypatch.setenv("AGENTS_TENANT_TOKENS", '{"team-a":"token-a","team-b":"token-b"}')
    response = client.post(
        f"/api/agents/{running_agent}/invoke",
        headers=auth("token-a"),
        json={"input": {"description": "Flight to Lisbon", "amount": 240}},
    )
    assert response.status_code == 200
    decision = response.json()["decision_id"]
    assert client.get("/api/decisions", headers=auth("token-b")).json() == []
    path = f"/api/decisions/{decision}/review"
    body = {"correct": True, "staff": "Dana"}
    assert client.post(path, headers=auth("token-b"), json=body).status_code == 404
    assert client.post(path, headers=auth("token-a"), json=body).status_code == 200
    value = client.get(f"/api/agents/{running_agent}/value", headers=auth("token-a")).json()
    assert value["reviewed"] == 1
    assert value["staff_saving"] == 2


def test_catalogue_pages(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Expense line coding" in page.text
    assert "no exam" in page.text
    agent = client.get("/agents/expense-coder")
    assert agent.status_code == 200
    assert "Break-even accuracy" in agent.text
    assert client.get("/agents/nobody").status_code == 404


def test_agent_page_with_exam(client, running_agent):
    page = client.get(f"/agents/{running_agent}")
    assert "Acceptance exam" in page.text
    assert "Projected monthly effect" in page.text
    assert "handed over, rightly" in page.text


def test_api_lists_agents(client):
    agents = client.get("/api/agents").json()
    assert agents[0]["name"] == "expense-coder"
    assert agents[0]["status"] == "no_exam"
    assert client.get("/api/agents/expense-coder").json()["exam"] is None


def test_invoke_is_refused_without_a_configured_token(client):
    r = client.post("/api/agents/expense-coder/invoke", json={"input": {}}, headers=auth())
    assert r.status_code == 503


def test_invoke_needs_the_right_token(client, monkeypatch):
    monkeypatch.setenv("AGENTS_API_TOKEN", "secret")
    url = "/api/agents/expense-coder/invoke"
    assert client.post(url, json={"input": {}}).status_code == 401
    assert client.post(url, json={"input": {}}, headers=auth("wrong")).status_code == 401


@pytest.mark.parametrize(
    "settings",
    [
        "invalid json",
        "[]",
        '{"team-a":3}',
        '{"team-a":""}',
        '{"":"token"}',
        '{"team-a":"shared","team-b":"shared"}',
        '{"team-a":"secret"}',
    ],
)
def test_invalid_tenant_credentials_cannot_select_a_company(client, monkeypatch, settings):
    monkeypatch.setenv("AGENTS_API_TOKEN", "secret")
    monkeypatch.setenv("AGENTS_TENANT_TOKENS", settings)
    response = client.get("/api/decisions", headers=auth("shared"))
    assert response.status_code == 503
    assert "invalid AGENTS_TENANT_TOKENS" in response.json()["detail"]


def test_tenant_only_setup_requires_a_matching_token(client, monkeypatch):
    monkeypatch.setenv("AGENTS_TENANT_TOKENS", '{"team-a":"token-a"}')
    assert client.get("/api/decisions", headers=auth("wrong")).status_code == 401
    assert client.get("/api/decisions").status_code == 401
    assert client.get("/api/decisions", headers=auth("token-a")).status_code == 200


def test_invoke_refuses_an_agent_that_is_not_running(client, monkeypatch):
    monkeypatch.setenv("AGENTS_API_TOKEN", "secret")
    r = client.post("/api/agents/expense-coder/invoke", json={"input": {}}, headers=auth())
    assert r.status_code == 409
    assert "no acceptance exam" in r.json()["detail"]


def test_invoke_a_running_agent(client, monkeypatch, running_agent):
    monkeypatch.setenv("AGENTS_API_TOKEN", "secret")
    body = {"input": {"description": "Flight to Lisbon", "amount": 240.0}}
    r = client.post(f"/api/agents/{running_agent}/invoke", json=body, headers=auth())
    assert r.status_code == 200, r.text
    assert r.json()["output"] == {"category": "travel"}
    assert r.json()["action_state"] is None
    assert r.json()["action_result"] is None
    assert r.json()["fell_back_to_person"] is False


def test_agent_failure_falls_back_to_a_person(client, monkeypatch, running_agent):
    monkeypatch.setenv("AGENTS_API_TOKEN", "secret")
    r = client.post(
        f"/api/agents/{running_agent}/invoke", json={"input": {"amount": 5}}, headers=auth()
    )
    assert r.status_code == 200
    assert r.json()["output"]["handover"] is True
    assert r.json()["fell_back_to_person"] is True


def test_run_a_scheduled_job_now(client, monkeypatch, running_agent):
    monkeypatch.setenv("AGENTS_API_TOKEN", "secret")
    r = client.post(f"/api/agents/{running_agent}/run", headers=auth())
    assert r.status_code == 200
    assert r.json()["cases"] == 2
    assert r.json()["handed_over"] == 1
    page = client.get(f"/agents/{running_agent}")
    assert "Recent scheduled runs" in page.text


def test_unknown_agent_is_404_even_with_a_token(client, monkeypatch):
    monkeypatch.setenv("AGENTS_API_TOKEN", "secret")
    assert client.post("/api/agents/nobody/run", headers=auth()).status_code == 404


@pytest.fixture
def action_agent(workspace, real_model):
    def create(autonomy="act_alone"):
        agent = make_agent(
            workspace,
            "expense-action",
            model="test-real",
            autonomy=autonomy,
            tools=["issue_refund"],
        )
        (agent / "action.py").write_text(
            "def build_action(payload, output):\n"
            "    if output.get('category') != 'travel':\n"
            "        return None\n"
            "    return {'tool': 'issue_refund', 'arguments': "
            "{'request_id': 'RR-9001', 'amount': 48}}\n"
        )
        report = run_exam(agent)
        save_report(report, acceptance_report_path(report.agent))
        return agent.name

    return create


@pytest.mark.parametrize("state", ["executed", "failed", "unknown", "pending", "duplicate"])
def test_invoke_reports_persisted_action_outcome(client, monkeypatch, action_agent, state):
    from agent_platform import tools

    name = action_agent("act_after_approval" if state == "pending" else "act_alone")
    monkeypatch.setenv("AGENTS_API_TOKEN", "secret")
    calls = []

    def write(**arguments):
        calls.append(arguments)
        if state == "unknown":
            raise TimeoutError("connection lost after sending write")
        return {"error": "write refused"} if state == "failed" else {"ok": True}

    monkeypatch.setattr(tools, "_function", lambda spec: write)
    url = f"/api/agents/{name}/invoke"
    body = {"input": {"description": "Flight to Lisbon", "amount": 240}}
    if state == "duplicate":
        assert client.post(url, headers=auth(), json=body).json()["action_state"] == "executed"
    response = client.post(url, headers=auth(), json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["output"] == {"category": "travel"}
    assert result["action_state"] == state
    assert result["fell_back_to_person"] is (state in ("failed", "unknown"))
    persisted = next(
        d
        for d in client.get("/api/decisions", headers=auth()).json()
        if d["id"] == result["decision_id"]
    )
    assert result["action_result"] == persisted["action_result"]
    if state == "executed":
        assert result["action_result"] == {"ok": True}
    elif state in ("failed", "unknown"):
        assert "error" in result["action_result"]
    else:
        assert result["action_result"] is None
    assert len(calls) == (0 if state == "pending" else 1)


@pytest.mark.parametrize("state", ["executed", "failed", "unknown"])
def test_job_counts_action_outcomes(client, monkeypatch, action_agent, state):
    from agent_platform import tools
    from agent_platform.runner import recent_runs

    name = action_agent()
    monkeypatch.setenv("AGENTS_API_TOKEN", "secret")

    def write(**arguments):
        if state == "unknown":
            raise TimeoutError("outcome uncertain")
        return {"error": "refused"} if state == "failed" else {"ok": True}

    monkeypatch.setattr(tools, "_function", lambda spec: write)
    response = client.post(f"/api/agents/{name}/run", headers=auth())
    assert response.status_code == 200
    result = response.json()
    assert result["action_counts"] == {state: 1}
    assert result["answered"] == (1 if state == "executed" else 0)
    assert result["handed_over"] == (1 if state == "executed" else 2)
    assert result["fell_back"] == (0 if state == "executed" else 1)
    assert result["cases"] == 2
    assert recent_runs(name)[0] == result
