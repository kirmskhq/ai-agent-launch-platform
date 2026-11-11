"""Where the platform keeps its files, and the settings that come from the environment.

Every path can be moved with an environment variable, so tests and deployments can point the
platform at their own folders without touching code:

    AGENTS_DIR           agent folders (default: ./agents)
    AGENTS_REPORTS_DIR   exam reports that decide what may run (default: ./reports)
    AGENTS_STATE_DIR     run history and other state written while running (default: ./var)
    AGENTS_CONFIG_DIR    platform.yaml and models.yaml (default: ./config)
"""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    agents: Path
    reports: Path
    state: Path
    config: Path


def paths() -> Paths:
    """Read on every call, so a changed environment takes effect at once."""
    return Paths(
        agents=Path(os.environ.get("AGENTS_DIR", ROOT / "agents")),
        reports=Path(os.environ.get("AGENTS_REPORTS_DIR", ROOT / "reports")),
        state=Path(os.environ.get("AGENTS_STATE_DIR", ROOT / "var")),
        config=Path(os.environ.get("AGENTS_CONFIG_DIR", ROOT / "config")),
    )


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env. Variables already set in the environment win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
