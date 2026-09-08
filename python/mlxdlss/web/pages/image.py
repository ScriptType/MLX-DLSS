"""Image page: source and result on the left, controls on the right."""
from __future__ import annotations

from nicegui import events, run, ui

from ..state import get_state
from . import ds
from .common import effect_editor, job_status_card, layout
from .preview import LivePreview


def result_card(job) -> None:
    with ds.card("Result") as box:
        with box.meta:
            ds.result_meta(job)
        ds.wipe_compare(f"/api/jobs/{job.id}/input", f"/api/jobs/{job.id}/download/0")


@ui.page("/")
def image_page(job: str | None = None) -> None:
    """``?job=ID`` opens a finished job's before/after comparison."""
    state = get_state()
    with layout("Image", "Adjust settings, preview the result, then export."):
        with ui.element("div").classes("mlxdlss-grid"):
            with ui.element("div").classes("mlxdlss-stack"):
                with ds.card("Source"):
                    zone = ds.dropzone(accept="image/*", title="Drop an image here, or click to browse", hint="PNG, JPEG, TIFF, WebP · up to 200 MB",
                                       on_upload=lambda e: on_upload(e), max_size=200_000_000)
                    picked = ui.element("div").classes("w-full").style("display: none")
                preview = LivePreview(state, "image")
                results = ui.element("div").classes("mlxdlss-stack")
                opened = state.store.get(job) if job else None
                if opened is not None and opened.kind != "image":
                    opened = None
                if opened is not None and opened.state == "done" and opened.outputs:
                    with results:
                        result_card(opened)
            with ui.element("div").classes("mlxdlss-stack"):
                chain = effect_editor("image", opened.effects if opened is not None else None, on_change=preview.request)
                preview.chain = chain
                ds.button("Export image", kind="primary", icon="play_arrow", large=True, on_click=lambda: submit())
                ds.button("Export all images", kind="secondary", on_click=lambda: submit(all_files=True)).bind_visibility_from(
                    preview.file_choice, "options", backward=lambda files: len(files) > 1)
        if opened is not None:
            preview.select(state.store.input_path(opened), opened.input_name)
            with picked:
                ds.file_row(opened.input_name, f"{preview.source.stat().st_size / 1e6:.1f} MB")
            picked.style("display: block")
            zone.classes(add="mlxdlss-drop-compact")

        async def on_upload(e: events.UploadEventArguments) -> None:
            try:
                await preview.upload(e.file)
            except ValueError as error:
                ui.notify(str(error), type="negative"); return
            picked.clear()
            with picked:
                ds.file_row(preview.name, f"{preview.source.stat().st_size / 1e6:.1f} MB")
            picked.style("display: block"); picked.update()
            zone.classes(add="mlxdlss-drop-compact")

        async def submit(all_files=False) -> None:
            if preview.source is None:
                ui.notify("Choose an image first.", type="warning"); return
            try:
                files = list(preview.files.items()) if all_files else [(str(preview.source), preview.name)]
                jobs = await run.io_bound(preview.create_jobs, chain(), {}, files)
            except ValueError as error:
                ui.notify(str(error), type="negative"); return
            for job in jobs:
                state.queue.submit(job)
                show_job(job)

        def show_job(job):
            with results:
                holder = ui.element("div").classes("mlxdlss-stack w-full")

                def show() -> None:
                    finished = state.store.get(job.id)
                    if finished is None or finished.state != "done":
                        return
                    with holder:
                        result_card(finished)

                job_status_card(job.id, on_done=show)
