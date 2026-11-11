import json
import shutil
from pathlib import Path

import pytest
import yaml

from agent_platform import gateway

FIXTURES = Path(__file__).parent / "fixtures"

MODELS = {
    "providers": {
        "testcloud": {
            "base_url": "http://127.0.0.1:9/v1",
            "api_key_env": "TESTCLOUD_KEY",
            "max_concurrency": 2,
            "retries": 2,
        },
        "mock": {"kind": "mock"},
    },
    "models": {
        "test-real": {
            "provider": "testcloud",
            "model": "vendor/test-real-1",
            "price_usd_per_1m": {"input": 1.0, "output": 2.0},
        },
        "test-other": {
            "provider": "testcloud",
            "model": "vendor/test-other-1",
            "price_usd_per_1m": {"input": 0.5, "output": 1.0},
        },
        "mock": {
            "provider": "mock",
            "model": "mock",
            "price_usd_per_1m": {"input": 0, "output": 0},
        },
    },
}


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch) -> Path:
    """Every test gets its own agents, reports, state and config folders."""
    agents = tmp_path / "agents"
    shutil.copytree(FIXTURES / "agents", agents)
    config = tmp_path / "config"
    config.mkdir()
    (config / "models.yaml").write_text(yaml.safe_dump(MODELS), encoding="utf-8")
    (config / "platform.yaml").write_text("currency: USD\n", encoding="utf-8")
    monkeypatch.setenv("AGENTS_DIR", str(agents))
    monkeypatch.setenv("AGENTS_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("AGENTS_STATE_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("AGENTS_CONFIG_DIR", str(config))
    for var in ("AGENTS_GATEWAY_URL", "AGENTS_API_TOKEN", "TESTCLOUD_KEY", "AGENTS_SCHEDULER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gateway, "_sleep", lambda s: None)
    gateway._clients.clear()
    gateway._slots.clear()
    return tmp_path


def holdout_answers() -> dict[str, str]:
    path = FIXTURES / "agents" / "expense-coder" / "evals" / "holdout.jsonl"
    answers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if "category" in case["expected"]:
            answers[case["input"]["description"]] = case["expected"]["category"]
    return answers


def oracle(messages: list[dict]) -> str:
    """A model that knows every holdout answer: lets the fixture pass its exam."""
    text = messages[-1]["content"]
    for description, category in holdout_answers().items():
        if description in text:
            return json.dumps({"category": category})
    return json.dumps({"category": "unclear"})


@pytest.fixture
def real_model(monkeypatch):
    """Stand in for a real provider: `test-real` answers like the oracle, with token usage."""
    monkeypatch.setenv("TESTCLOUD_KEY", "test-key")

    def reply(alias, spec, provider, messages, temperature, max_tokens):
        return oracle(messages), 1000, 50

    monkeypatch.setattr(gateway, "_remote_reply", reply)


def make_agent(workspace: Path, name: str, **changes) -> Path:
    """Copy the fixture agent under a new name with passport fields changed."""
    src = workspace / "agents" / "expense-coder"
    dst = workspace / "agents" / name
    shutil.copytree(src, dst)
    passport = yaml.safe_load((dst / "agent.yaml").read_text(encoding="utf-8"))
    passport["name"] = name
    for key, value in changes.items():
        if isinstance(value, dict):
            passport[key] = {**passport.get(key, {}), **value}
        else:
            passport[key] = value
    (dst / "agent.yaml").write_text(yaml.safe_dump(passport), encoding="utf-8")
    set_proposal(dst, name=name)
    return dst


def set_proposal(agent_dir: Path, **changes) -> None:
    """Change top-level fields of an agent's proposal.yaml (None removes the field)."""
    path = agent_dir / "proposal.yaml"
    proposal = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            proposal.pop(key, None)
        elif isinstance(value, dict) and isinstance(proposal.get(key), dict):
            proposal[key] = {**proposal[key], **value}
        else:
            proposal[key] = value
    path.write_text(yaml.safe_dump(proposal), encoding="utf-8")
