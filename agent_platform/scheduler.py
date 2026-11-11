"""Scheduler: starts scheduled jobs of running agents. All schedules are in UTC.

Checks every 20 seconds. An agent starts at most once per minute it is due, and never while
its previous run is still going.
"""

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from croniter import croniter

from agent_platform.registry import records
from agent_platform.runner import run_job

log = logging.getLogger("agent_platform.scheduler")
TICK_S = 20


def is_due(schedule: str, now: datetime) -> bool:
    if now.tzinfo is None:
        raise ValueError("now must carry a time zone")
    minute = now.astimezone(UTC).replace(second=0, microsecond=0, tzinfo=None)
    return croniter.match(schedule, minute)


class Scheduler:
    def __init__(self, run: Callable[..., dict] = run_job):
        self._run_job = run
        self._fired: dict[str, str] = {}
        self._running: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self, now: datetime | None = None) -> list[str]:
        """Start every agent due at `now`. Returns the names started."""
        now = now or datetime.now(UTC)
        stamp = now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M")
        started = []
        for agent in records():
            schedule = agent.passport.schedule if agent.passport else None
            if not agent.runs or not schedule or not is_due(schedule, now):
                continue
            with self._lock:
                if self._fired.get(agent.name) == stamp or agent.name in self._running:
                    continue
                self._fired[agent.name] = stamp
                self._running.add(agent.name)
            threading.Thread(target=self._run, args=(agent.name,), daemon=True).start()
            started.append(agent.name)
        return started

    def is_running(self, name: str) -> bool:
        with self._lock:
            return name in self._running

    def _run(self, name: str) -> None:
        try:
            entry = self._run_job(name, trigger="schedule")
            log.info("%s: %s cases, %s handed over", name, entry["cases"], entry["handed_over"])
        except Exception:
            log.exception("%s: scheduled run failed", name)
        finally:
            with self._lock:
                self._running.discard(name)

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception:
                    log.exception("scheduler tick failed")
                self._stop.wait(TICK_S)

        self._thread = threading.Thread(target=loop, name="scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
