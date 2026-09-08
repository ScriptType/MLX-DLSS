"""Live still-frame controls shared by image and video pages."""
from __future__ import annotations

import asyncio
import tempfile
import threading
import uuid
from pathlib import Path

from nicegui import ui

from ..effects import media_kind
from ..preview import LatestPreview, PreviewSession
from . import ds


class LivePreview:
    def __init__(self, state, kind):
        self.state, self.kind = state, kind
        self.source = None
        self.name = ""
        self.files = {}
        self.files_lock = threading.Lock()
        self.chain = lambda: []
        self.result = None
        self.updating = False
        self.directory = tempfile.TemporaryDirectory(prefix="mlxdlss-preview-")
        self.session = PreviewSession(state.runner)
        state.previews.add(self.session)
        self.worker = LatestPreview(lambda request: self.session.render(*request), self.show, self.error,
                                    busy=lambda: any(j.state in {"queued", "running"} for j in state.store.list()))
        with ds.card("Live preview") as card:
            self.card = card
            card.classes("mlxdlss-live-card")
            with card.meta:
                self.view = ui.toggle(["Original", "Live"], value="Live", on_change=lambda _: self.display()).props(
                    "unelevated no-caps dense toggle-color=primary").classes("mlxdlss-seg")
            self.file_choice = ui.select({}, label="Selected file", on_change=lambda e: self.select(e.value, self.files[e.value])
                                         if e.value and not self.updating else None).props("outlined dense").classes("w-full")
            self.file_choice.set_visibility(False)
            self.picture = ui.image().props('fit=contain no-spinner').classes("mlxdlss-preview").style("min-height: 220px")
            if kind == "video":
                with ui.row().classes("w-full items-center no-wrap gap-2"):
                    ds.icon_button("skip_previous", tooltip="Previous frame", on_click=lambda: self.step(-1))
                    self.timeline = ui.slider(min=0, max=0.001, step=1 / 30, value=0).props('aria-label="Preview time"').classes("flex-1")
                    self.timeline.on_value_change(lambda _: self.request())
                    ds.icon_button("skip_next", tooltip="Next frame", on_click=lambda: self.step(1))
                self.position = ui.label().classes("mlxdlss-muted mlxdlss-mono")
            self.status = ui.label("Choose a file to preview settings.").classes("mlxdlss-muted mlxdlss-small")
            if kind == "video":
                ui.label("Up to 3 preceding frames. Export uses the full history and frame generation.").classes("mlxdlss-muted mlxdlss-small")
        ui.context.client.on_delete(self.close)

    async def upload(self, file):
        if media_kind(file.name) != self.kind:
            raise ValueError(f"Choose an {self.kind} file")
        target = Path(self.directory.name) / (uuid.uuid4().hex + Path(file.name).suffix.lower())
        await file.save(target)
        self.select(target, Path(file.name).name)

    def select(self, source, name):
        self.source, self.name = Path(source), name
        self.files[str(source)] = name
        self.updating = True
        self.file_choice.set_options(self.files, value=str(source))
        self.file_choice.set_visibility(len(self.files) > 1)
        self.result = None
        self.picture.set_source("")
        if self.kind == "video":
            self.timeline.set_value(0)
        self.updating = False
        self.request()

    def create_jobs(self, effects, output, files):
        with self.files_lock:
            if self.worker.closed:
                raise ValueError("The page was closed before export started")
            return [self.state.store.create(name, effects, source=Path(path), output_options=output) for path, name in files]

    def request(self):
        if self.source is None or self.updating:
            return
        self.status.classes(remove="mlxdlss-bad")
        busy = any(j.state in {"queued", "running"} for j in self.state.store.list())
        self.status.set_text("Waiting for export…" if busy else "Updating…")
        self.worker.submit((self.source, self.kind, self.chain(), float(self.timeline.value) if self.kind == "video" else 0))

    def show(self, result):
        self.result = result
        self.updating = True
        if self.kind == "video":
            self.timeline._props.update(max=max(0.001, result["duration"] - result["frameInterval"]), step=result["frameInterval"])
            self.timeline.set_value(result["time"])
            self.timeline.update()
            self.position.set_text(f'{result["time"]:.3f} / {result["duration"]:.2f} s')
        self.updating = False
        temporal = f' · Temporal · {result["historyFrames"]} preceding frames' if result["historyFrames"] else ""
        self.status.set_text(f'Output: {result["width"]} × {result["height"]}{temporal} · {result["elapsedSeconds"]:.2f} s')
        self.display()

    def display(self):
        if self.result:
            key = "original" if self.view.value == "Original" else "processed"
            self.picture.set_source("data:image/png;base64," + self.result[key])

    def error(self, message):
        self.status.set_text(message)
        self.status.classes(add="mlxdlss-bad")

    def step(self, direction):
        if self.result:
            self.timeline.set_value(max(0, min(self.result["duration"] - self.result["frameInterval"],
                                               self.result["time"] + direction * self.result["frameInterval"])))

    async def close(self):
        self.worker.close()
        await asyncio.to_thread(self.session.close)
        if self.worker.task is not None:
            await self.worker.task
        with self.files_lock:
            self.directory.cleanup()
