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
