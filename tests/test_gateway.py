import threading
import time
from types import SimpleNamespace

import httpx
import openai
import pytest
import yaml

from agent_platform import gateway
from agent_platform.gateway import (
    BadModelReply,
    GatewayError,
    ModelUnavailable,
    chat,
    chat_json,
    mock_model,
    parse_json_object,
    track,
    use_model,
)

REQUEST = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions")


def rate_limited() -> openai.RateLimitError:
    response = httpx.Response(429, request=REQUEST)
    return openai.RateLimitError("slow down", response=response, body=None)


def bad_request() -> openai.BadRequestError:
    return openai.BadRequestError("bad", response=httpx.Response(400, request=REQUEST), body=None)


def completion(text: str, prompt_tokens: int = 1_000_000, completion_tokens: int = 500_000):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def fake(monkeypatch):
    def install(replies, via_proxy=False):
        client = FakeClient(replies)
        monkeypatch.setattr(gateway, "_client", lambda name, provider: (client, via_proxy))
        return client

    return install


def test_chat_needs_a_selected_model():
    with pytest.raises(GatewayError, match="no model selected"):
        chat("hello")


def test_unknown_model():
    with use_model("nope"), pytest.raises(GatewayError, match="not in config"):
        chat("hello")


def test_mock_model_and_ledger():
    with use_model("mock"), mock_model(lambda m: "pong"), track() as ledger:
        assert chat("ping", system="be brief") == "pong"
    assert len(ledger.calls) == 1
    assert ledger.cost == 0
    assert ledger.calls[0].model == "mock"


def test_mock_without_responder():
    with use_model("mock"), track() as ledger, pytest.raises(GatewayError, match="no mock"):
        chat("ping")
    assert ledger.failed_calls == 1


def test_cost_in_platform_currency(fake, workspace):
    fake([completion("ok")])
    with use_model("test-real"), track() as ledger:
        chat("hi")
    # 1M prompt tokens at 1.0 USD + 0.5M completion tokens at 2.0 USD
    assert ledger.cost == pytest.approx(2.0)

    (workspace / "config" / "platform.yaml").write_text(
        "currency: EUR\nusd_rate: 0.5\n", encoding="utf-8"
    )
    fake([completion("ok")])
    with use_model("test-real"), track() as ledger:
        chat("hi")
    assert ledger.cost == pytest.approx(1.0)


def test_retries_transient_errors(fake):
    client = fake([rate_limited(), rate_limited(), completion("finally")])
    with use_model("test-real"):
        assert chat("hi") == "finally"
    assert len(client.requests) == 3
    assert client.requests[0]["model"] == "vendor/test-real-1"


def test_backoff_doubles_up_to_the_provider_ceiling(fake, monkeypatch):
    waits = []
    monkeypatch.setattr(gateway, "_sleep", waits.append)
    monkeypatch.setattr(gateway.random, "random", lambda: 1.0)
    fake([rate_limited(), rate_limited(), completion("ok")])
    with use_model("test-real"):
        chat("hi")
    assert waits == [1, 2]
    assert gateway._backoff(10, 60) == 60


def limited(seconds: str) -> openai.RateLimitError:
    response = httpx.Response(429, request=REQUEST, headers={"Retry-After": seconds})
    return openai.RateLimitError("slow down", response=response, body=None)


def test_waits_as_long_as_the_provider_asks(fake, monkeypatch):
    waits = []
    monkeypatch.setattr(gateway, "_sleep", waits.append)
    fake([limited("7"), completion("ok")])
    with use_model("test-real"):
        chat("hi")
    assert waits[0] >= 7


def test_a_long_quota_wait_fails_at_once(fake):
    client = fake([limited("7200"), completion("never")])
    with use_model("test-real"), pytest.raises(ModelUnavailable, match="wait 7200s"):
        chat("hi")
    assert len(client.requests) == 1


def test_requests_are_spaced_to_the_per_minute_limit(monkeypatch):
    waits = []
    monkeypatch.setattr(gateway, "_sleep", waits.append)
    clock = iter([100.0, 100.0, 100.0])
    monkeypatch.setattr(gateway.time, "monotonic", lambda: next(clock))
    for _ in range(3):
        gateway._pace("p", 20)
    assert waits == pytest.approx([3.0, 6.0])


def test_timeouts_are_retried(fake):
    client = fake([openai.APITimeoutError(request=REQUEST), completion("late but fine")])
    with use_model("test-real"):
        assert chat("hi") == "late but fine"
    assert len(client.requests) == 2


def test_gives_up_after_retries(fake):
    client = fake([rate_limited()] * 3)  # retries: 2 in the test config -> 3 attempts
    with use_model("test-real"), track() as ledger, pytest.raises(ModelUnavailable):
        chat("hi")
    assert len(client.requests) == 3
    assert ledger.failed_calls == 1


def test_does_not_retry_a_bad_request(fake):
    client = fake([bad_request(), completion("never")])
    with use_model("test-real"), pytest.raises(GatewayError, match="BadRequestError"):
        chat("hi")
    assert len(client.requests) == 1


def test_model_options_go_with_the_request(fake, workspace):
    path = workspace / "config" / "models.yaml"
    models = yaml.safe_load(path.read_text(encoding="utf-8"))
    models["models"]["test-real"]["options"] = {"reasoning_effort": "low"}
    path.write_text(yaml.safe_dump(models), encoding="utf-8")
    client = fake([completion("ok")])
    with use_model("test-real"):
        chat("hi")
    assert client.requests[0]["extra_body"] == {"reasoning_effort": "low"}


def test_proxy_uses_the_alias(fake):
    client = fake([completion("ok")], via_proxy=True)
    with use_model("test-real"):
        chat("hi")
    assert client.requests[0]["model"] == "test-real"


def test_missing_key():
    with use_model("test-real"), pytest.raises(GatewayError, match="set TESTCLOUD_KEY"):
        chat("hi")


def test_concurrency_limit(monkeypatch):
    monkeypatch.setenv("TESTCLOUD_KEY", "k")
    in_flight = peak = 0
    lock = threading.Lock()

    def create(**request):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return completion("ok", 1, 1)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(gateway, "_client", lambda name, provider: (client, False))

    def call():
        with use_model("test-real"):
            chat("hi")

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 2  # max_concurrency in the test config


def test_chat_json_asks_again_once():
    replies = iter(["sure, here it is", '{"ok": true}'])
    with use_model("mock"), mock_model(lambda m: next(replies)), track() as ledger:
        assert chat_json("give json") == {"ok": True}
    assert len(ledger.calls) == 2


def test_chat_json_gives_up_after_one_retry():
    with use_model("mock"), mock_model(lambda m: "no"), pytest.raises(BadModelReply):
        chat_json("give json")


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Answer: {"a": 1} hope that helps'],
)
def test_parse_json_object(text):
    assert parse_json_object(text) == {"a": 1}


def test_parse_json_object_rejects_lists():
    with pytest.raises(BadModelReply):
        parse_json_object("[1, 2]")
