"""The one door between agents and language models.

Agents call `chat()` or `chat_json()` and nothing else. They do not know the provider, the
key or even the model: the platform selects the passport's model with `use_model()` before
running an agent, and changing models is a change to the passport, not to agent code.

Every call is written to the open ledgers (`track()`) with tokens, latency and cost in the
platform currency. The exam reads cost per case from them, the runner reads cost per run.

Failures are split in two:
- `GatewayError`: the model could not be used (no key, unknown model, provider down after all
  retries). This is never the agent's fault, and the exam reports it as an infrastructure
  failure rather than a wrong answer.
- `BadModelReply`: the model answered, but not with what was asked (for example no JSON).
  That is part of the agent's quality and is scored as such.

With AGENTS_GATEWAY_URL set, every request goes to a LiteLLM proxy instead, which holds the
provider keys; the agent host then only needs the proxy address and AGENTS_GATEWAY_KEY.
"""

import json
import os
import random
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import openai
from openai import OpenAI

from agent_platform.config import ModelSpec, Price, Provider, models_config, platform_config

Messages = list[dict[str, str]]
MockModel = Callable[[Messages], str]

TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class GatewayError(RuntimeError):
    """The model could not be used. Not the agent's fault."""


class ModelUnavailable(GatewayError):
    """The provider kept failing after every retry."""


class BadModelReply(ValueError):
    """The model answered, but not in the form the agent asked for."""


@dataclass(frozen=True)
class ModelCall:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency_s: float
    error: str | None = None


@dataclass
class Ledger:
    calls: list[ModelCall] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)

    @property
    def failed_calls(self) -> int:
        return sum(c.error is not None for c in self.calls)


_current_model: ContextVar[str | None] = ContextVar("current_model", default=None)
_open_ledgers: ContextVar[tuple[Ledger, ...]] = ContextVar("open_ledgers", default=())
_mock_model: ContextVar[MockModel | None] = ContextVar("mock_model", default=None)

_clients: dict[str, OpenAI] = {}
_slots: dict[str, threading.BoundedSemaphore] = {}
_next_start: dict[str, float] = {}
_lock = threading.Lock()
_sleep = time.sleep  # replaced in tests


@contextmanager
def use_model(alias: str) -> Iterator[None]:
    """Select the model for everything called inside the block."""
    token = _current_model.set(alias)
    try:
        yield
    finally:
        _current_model.reset(token)


@contextmanager
def track() -> Iterator[Ledger]:
    """Collect every model call made inside the block. Nested blocks all see the calls."""
    ledger = Ledger()
    token = _open_ledgers.set((*_open_ledgers.get(), ledger))
    try:
        yield ledger
    finally:
        _open_ledgers.reset(token)


@contextmanager
def mock_model(responder: MockModel) -> Iterator[None]:
    """Answers for the `mock` model: tests and runs without a key. Costs nothing."""
    token = _mock_model.set(responder)
    try:
        yield
    finally:
        _mock_model.reset(token)


def current_model() -> str:
    alias = _current_model.get()
    if alias is None:
        raise GatewayError("no model selected; the platform selects the passport's model")
    return alias


def chat(
    prompt: str | Messages,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Send a chat request to the selected model and return the reply text."""
    alias = current_model()
    cfg = models_config()
    if alias not in cfg.models:
        raise GatewayError(f"model {alias!r} is not in config/models.yaml")
    spec = cfg.models[alias]
    provider = cfg.providers[spec.provider]
    messages = _messages(prompt, system)

    started = time.perf_counter()
    try:
        if provider.kind == "mock":
            text, prompt_tokens, completion_tokens = _mock_reply(messages)
        else:
            text, prompt_tokens, completion_tokens = _remote_reply(
                alias, spec, provider, messages, temperature, max_tokens
            )
    except GatewayError as e:
        _record(ModelCall(alias, 0, 0, 0.0, time.perf_counter() - started, error=str(e)))
        raise
    cost = _cost(spec.price_usd_per_1m, prompt_tokens, completion_tokens)
    _record(ModelCall(alias, prompt_tokens, completion_tokens, cost, time.perf_counter() - started))
    return text


def chat_json(prompt: str | Messages, *, system: str | None = None, **kwargs) -> dict:
    """`chat()` that must return a JSON object. Asks once more if the first reply is not one."""
    text = chat(prompt, system=system, **kwargs)
    try:
        return parse_json_object(text)
    except BadModelReply:
        retry = [
            *_messages(prompt, None),
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Reply with one valid JSON object and nothing else."},
        ]
        return parse_json_object(chat(retry, system=system, **kwargs))


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict:
    """Read a JSON object from a reply: bare, in a ```json fence, or inside other text."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    for attempt in (candidate, candidate[candidate.find("{") : candidate.rfind("}") + 1]):
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise BadModelReply(f"expected a JSON object, got: {text[:200]!r}")


def _messages(prompt: str | Messages, system: str | None) -> Messages:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    if system:
        messages = [{"role": "system", "content": system}, *messages]
    return messages


def _record(call: ModelCall) -> None:
    for ledger in _open_ledgers.get():
        ledger.calls.append(call)


def _cost(price: Price, prompt_tokens: int, completion_tokens: int) -> float:
    usd = (prompt_tokens * price.input + completion_tokens * price.output) / 1_000_000
    return usd * platform_config().usd_rate


def _mock_reply(messages: Messages) -> tuple[str, int, int]:
    responder = _mock_model.get()
    if responder is None:
        raise GatewayError("the mock model was called, but no mock responder is set")
    text = responder(messages)
    # A rough token count, so reports show the shape of the cost even on the mock.
    return text, sum(len(m["content"]) for m in messages) // 4, len(text) // 4


def _slot(provider_name: str, limit: int) -> threading.BoundedSemaphore:
    with _lock:
        if provider_name not in _slots:
            _slots[provider_name] = threading.BoundedSemaphore(limit)
        return _slots[provider_name]


def _client(provider_name: str, provider: Provider) -> tuple[OpenAI, bool]:
    """An OpenAI-protocol client, and whether it points at the LiteLLM proxy."""
    proxy = os.environ.get("AGENTS_GATEWAY_URL")
    key = f"proxy:{proxy}" if proxy else provider_name
    with _lock:
        if key not in _clients:
            if proxy:
                api_key = os.environ.get("AGENTS_GATEWAY_KEY", "none")
                base_url = proxy
            else:
                api_key = os.environ.get(provider.api_key_env or "", "")
                if not api_key:
                    raise GatewayError(
                        f"no API key for {provider_name}: set {provider.api_key_env} "
                        "(or AGENTS_GATEWAY_URL to go through LiteLLM)"
                    )
                base_url = provider.base_url
            _clients[key] = OpenAI(
                base_url=base_url, api_key=api_key, timeout=provider.timeout_s, max_retries=0
            )
        return _clients[key], bool(proxy)


def has_key(provider: Provider) -> bool:
    if provider.kind == "mock" or os.environ.get("AGENTS_GATEWAY_URL"):
        return True
    return bool(os.environ.get(provider.api_key_env or ""))


def _backoff(attempt: int, ceiling: float) -> float:
    return min(2**attempt, ceiling) * (0.5 + random.random() / 2)


def _retry_after(error: Exception) -> float | None:
    """The provider's own Retry-After, in seconds, if it sent one."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    for name, scale in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(name)
        if value is not None:
            try:
                return float(value) / scale
            except ValueError:
                return None
    return None


def _pace(provider_name: str, per_minute: int | None) -> None:
    """Wait for this provider's next free slot in its per-minute budget."""
    if not per_minute:
        return
    gap = 60 / per_minute
    with _lock:
        now = time.monotonic()
        start = max(now, _next_start.get(provider_name, now))
        _next_start[provider_name] = start + gap
    if start > now:
        _sleep(start - now)


def _remote_reply(
    alias: str,
    spec: ModelSpec,
    provider: Provider,
    messages: Messages,
    temperature: float,
    max_tokens: int | None,
) -> tuple[str, int, int]:
    client, via_proxy = _client(spec.provider, provider)
    # The proxy knows models by their platform alias; providers know them by their own id.
    request = {
        "model": alias if via_proxy else spec.model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if spec.options:
        request["extra_body"] = dict(spec.options)

    attempts = provider.retries + 1
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with _slot(spec.provider, provider.max_concurrency):
                _pace(spec.provider, provider.requests_per_minute)
                response = client.chat.completions.create(**request)
        except TRANSIENT_ERRORS as e:
            last = e
            asked = _retry_after(e)
            if asked is not None and asked > provider.max_wait_s:
                # A quota window that resets in hours: waiting is pointless, and retrying
                # early only gets the account paused.
                raise ModelUnavailable(
                    f"{alias}: the provider asks to wait {asked:.0f}s, longer than the "
                    f"{provider.max_wait_s:.0f}s allowed: {e}"
                ) from e
            if attempt + 1 < attempts:
                # Never earlier than the provider asks for.
                _sleep(max(asked or 0.0, _backoff(attempt, provider.max_backoff_s)))
            continue
        except openai.APIError as e:
            raise GatewayError(f"{alias}: {type(e).__name__}: {e}") from e
        text = response.choices[0].message.content or ""
        usage = response.usage
        return (
            text,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
    raise ModelUnavailable(
        f"{alias}: still failing after {attempts} attempts ({type(last).__name__}: {last})"
    ) from last
