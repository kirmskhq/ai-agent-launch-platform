"""Platform settings (config/platform.yaml) and the models agents may use (config/models.yaml).

An agent's passport names a model by its alias. Provider, address, key variable and price live
here, so changing an agent's model is a one-line change in its passport, and adding a provider
is a change in one file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_platform.settings import paths


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlatformConfig(_Strict):
    currency: str = Field(
        default="USD",
        pattern=r"^[A-Z]{3}$",
        description="ISO 4217 code. Every amount in passports and reports is in this currency.",
    )
    usd_rate: float = Field(
        default=1.0,
        gt=0,
        description="Units of `currency` per 1 USD. Model prices are kept in USD, as providers "
        "publish them, and converted with this rate.",
    )
    hours_per_staff_month: float = Field(
        default=160, gt=0, description="Working hours in one full-time staff month."
    )

    @model_validator(mode="after")
    def _usd_is_one(self) -> "PlatformConfig":
        if self.currency == "USD" and self.usd_rate != 1.0:
            raise ValueError("usd_rate must be 1.0 when currency is USD")
        return self


class Price(_Strict):
    input: float = Field(ge=0, description="USD per 1M prompt tokens")
    output: float = Field(ge=0, description="USD per 1M completion tokens")


class Provider(_Strict):
    kind: Literal["openai_compatible", "mock"] = "openai_compatible"
    base_url: str | None = None
    api_key_env: str | None = None
    max_concurrency: int = Field(default=4, ge=1, description="Requests in flight at once")
    retries: int = Field(default=4, ge=0, description="Retries on rate limits and outages")
    timeout_s: float = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _remote_needs_address(self) -> "Provider":
        if self.kind == "openai_compatible" and not (self.base_url and self.api_key_env):
            raise ValueError("an openai_compatible provider needs base_url and api_key_env")
        return self


class ModelSpec(_Strict):
    provider: str
    model: str = Field(description="The provider's own model id")
    price_usd_per_1m: Price


class ModelsConfig(_Strict):
    providers: dict[str, Provider]
    models: dict[str, ModelSpec]

    @model_validator(mode="after")
    def _providers_exist(self) -> "ModelsConfig":
        for alias, spec in self.models.items():
            if spec.provider not in self.providers:
                raise ValueError(f"model {alias!r} uses unknown provider {spec.provider!r}")
        return self

    def is_mock(self, alias: str) -> bool:
        spec = self.models.get(alias)
        return spec is not None and self.providers[spec.provider].kind == "mock"


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=16)
def _platform(path: str, _mtime: int) -> PlatformConfig:
    return PlatformConfig.model_validate(_read_yaml(Path(path)))


@lru_cache(maxsize=16)
def _models(path: str, _mtime: int) -> ModelsConfig:
    return ModelsConfig.model_validate(_read_yaml(Path(path)))


def platform_config() -> PlatformConfig:
    path = paths().config / "platform.yaml"
    if not path.is_file():
        return PlatformConfig()
    return _platform(str(path), path.stat().st_mtime_ns)


def models_config() -> ModelsConfig:
    path = paths().config / "models.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no model list at {path}")
    return _models(str(path), path.stat().st_mtime_ns)
