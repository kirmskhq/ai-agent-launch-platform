"""LiteLLM proxy config generated from config/models.yaml, so the model list lives in one place.

LiteLLM reads provider keys from its own environment (os.environ/...). Agent hosts only know
the proxy address and its key. CI fails if deploy/litellm.yaml is out of date.
"""

from pathlib import Path

import yaml

from agent_platform.config import models_config
from agent_platform.settings import ROOT

OUT = ROOT / "deploy" / "litellm.yaml"
HEADER = "# Generated from config/models.yaml by `agents litellm-config`. Do not edit by hand.\n"


def build() -> str:
    cfg = models_config()
    model_list = []
    for alias, spec in cfg.models.items():
        provider = cfg.providers[spec.provider]
        if provider.kind == "mock":
            continue
        model_list.append(
            {
                "model_name": alias,
                "litellm_params": {
                    "model": f"openai/{spec.model}",
                    "api_base": provider.base_url,
                    "api_key": f"os.environ/{provider.api_key_env}",
                    "max_parallel_requests": provider.max_concurrency,
                    "timeout": provider.timeout_s,
                    **spec.options,
                },
            }
        )
    doc = {
        "model_list": model_list,
        "litellm_settings": {"drop_params": True, "num_retries": 2},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
    }
    return HEADER + yaml.safe_dump(doc, sort_keys=False)


def write(path: Path = OUT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(), encoding="utf-8", newline="\n")


def is_current(path: Path = OUT) -> bool:
    return path.is_file() and path.read_text(encoding="utf-8") == build()
