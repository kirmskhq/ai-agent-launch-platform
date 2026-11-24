"""Who is calling: service tokens run agents, personal staff tokens review and approve.

    AGENTS_API_TOKEN      service token of the default company
    AGENTS_TENANT_TOKENS  {"team-a": "<token>"}: one service token per further company
    AGENTS_STAFF_TOKENS   {"Dana": {"token": "<token>", "role": "approver", "tenant": "team-a"}}
                          one personal token per person; tenant defaults to the default company

Roles and what they may do:

    service    run agents and scheduled jobs, read decisions and measured value
    reviewer   read decisions and measured value, mark answers right or wrong
    approver   everything a reviewer may do, and approve or reject queued actions

A review or an approval is recorded under the name its token belongs to, never a name typed into
the request, so service tokens, which belong to systems rather than people, can do neither.

The settings are read and checked once, when the service starts. A mistake stops the start
instead of surfacing at the first request; restart the service after changing a token.
"""

import hmac
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Role = Literal["service", "reviewer", "approver"]
Permission = Literal["run", "read", "review", "approve"]
DEFAULT_TENANT = "default"
STAFF_ROLES = ("reviewer", "approver")
PERMISSIONS: dict[Permission, frozenset[Role]] = {
    "run": frozenset({"service"}),
    "read": frozenset({"service", "reviewer", "approver"}),
    "review": frozenset({"reviewer", "approver"}),
    "approve": frozenset({"approver"}),
}


class CredentialsError(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    tenant: str
    role: Role
    name: str | None = None  # the person behind a staff token; None for a service token

    def can(self, permission: Permission) -> bool:
        return self.role in PERMISSIONS[permission]


@dataclass(frozen=True)
class Credentials:
    tokens: tuple[tuple[str, Principal], ...] = ()

    @property
    def configured(self) -> bool:
        return bool(self.tokens)

    def identify(self, token: str) -> Principal | None:
        """The owner of `token`, comparing against every known token in constant time."""
        found = None
        for known, principal in self.tokens:
            if hmac.compare_digest(token.encode(), known.encode()):
                found = principal
        return found

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Credentials":
        tokens: list[tuple[str, Principal]] = []
        default = env.get("AGENTS_API_TOKEN", "")
        if default.strip():
            tokens.append((default, Principal(DEFAULT_TENANT, "service")))

        tenants = _json_object(env, "AGENTS_TENANT_TOKENS")
        for tenant, token in tenants.items():
            if not _text(tenant) or not _text(token):
                raise CredentialsError(
                    "AGENTS_TENANT_TOKENS: tenant names and tokens must be nonempty strings"
                )
            tokens.append((token, Principal(tenant, "service")))

        companies = {DEFAULT_TENANT, *tenants}
        names: set[str] = set()
        for name, entry in _json_object(env, "AGENTS_STAFF_TOKENS").items():
            where = f"AGENTS_STAFF_TOKENS[{name!r}]"
            if not _text(name):
                raise CredentialsError("AGENTS_STAFF_TOKENS: staff names must be nonempty strings")
            if not isinstance(entry, dict):
                raise CredentialsError(f"{where} must be an object with a token and a role")
            if unknown := sorted(set(entry) - {"token", "role", "tenant"}):
                raise CredentialsError(f"{where}: unknown fields {', '.join(unknown)}")
            token, role = entry.get("token"), entry.get("role")
            tenant = entry.get("tenant", DEFAULT_TENANT)
            if not _text(token):
                raise CredentialsError(f"{where}: token must be a nonempty string")
            if role not in STAFF_ROLES:
                raise CredentialsError(f"{where}: role must be reviewer or approver")
            if not _text(tenant) or tenant not in companies:
                raise CredentialsError(f"{where}: unknown tenant {tenant!r}")
            if name.strip() in names:
                raise CredentialsError(f"{where}: the name is used twice")
            names.add(name.strip())
            tokens.append((token, Principal(tenant, role, name.strip())))

        values = [token for token, _ in tokens]
        if len(values) != len(set(values)):
            raise CredentialsError(
                "every token must be distinct: the default company, each tenant and each "
                "person need their own"
            )
        return cls(tuple(tokens))


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _json_object(env: Mapping[str, str], name: str) -> dict:
    raw = env.get(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CredentialsError(f"{name} is not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise CredentialsError(f"{name} must be a JSON object")
    return value
