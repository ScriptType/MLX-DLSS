"""The process-wide wiring: settings, job store, queue and the runner behind them."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from weakref import WeakSet

from .jobs import JobQueue, JobStore
from .runners import JobRunner, ModelCache
from .settings import Settings


class WebState:
    def __init__(self, settings: Settings | None = None, *, root: Path | None = None, runner=None):
        self.settings = settings or Settings.load()
        if root is not None:
            self.settings.root = str(root)
        self.settings.outputs.mkdir(parents=True, exist_ok=True)
        self.cache = ModelCache()
        self.store = JobStore(self.settings.outputs)
        self.runner = runner or JobRunner(lambda: self.settings, self.cache)
        self.queue = JobQueue(self.store, self.runner)
        self.previews = WeakSet()

    def update_settings(self, **changes) -> None:
        updated = replace(self.settings, **changes)
        if updated.outputs != self.settings.outputs:
            if any(job.state in {"queued", "running"} for job in self.store.list()):
                raise ValueError("Finish or cancel queued jobs before changing the output folder")
            store = JobStore(updated.outputs)
            self.queue.close()
            self.store = store
            self.queue = JobQueue(self.store, self.runner)
        self.settings = updated
        self.settings.save()

    def close(self) -> None:
        for job in self.store.list():
            if job.state in {"queued", "running"}:
                self.queue.cancel(job.id)
        for preview in list(self.previews):
            preview.close()
        self.queue.close()


STATE: WebState | None = None


def get_state() -> WebState:
    global STATE
    if STATE is None:
        STATE = WebState()
    return STATE
