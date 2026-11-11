# AI agent launch platform

A platform that takes AI agents from a business department's process to production.
A department proposes an agent with a passport: the work it replaces, the baseline, the cost of
an error and the metric it should move. The agent goes live only after it passes an exam on
real past cases, and its value is measured against the baseline.

Work in progress. The full guide comes with the last stage.

```bash
uv sync
uv run pytest
uv run agents --help
```
