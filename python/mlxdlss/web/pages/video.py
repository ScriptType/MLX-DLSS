"""Video page: source and preview on the left, the effect chain on the right."""
from __future__ import annotations

from nicegui import events, run, ui

from ..state import get_state
from . import ds
from .common import effect_editor, job_status_card, layout, output_editor
from .preview import LivePreview


def result_card(job) -> None:
    """The converted clip; the side-by-side comparison with the original is a view, not the product."""
    result_url = f"/api/jobs/{job.id}/output/0"
    with ds.card("Result") as box:
        with box.meta:
            if job.preview:
                view = ui.toggle({"result": "Result", "compare": "Side by side"}, value="result").props("unelevated no-caps dense toggle-color=primary").classes("mlxdlss-seg")
            ds.result_meta(job)
        player = ui.video(result_url).classes("mlxdlss-preview")
        note = ui.label(f"{job.input_name} → {job.outputs[0]}").classes("mlxdlss-muted mlxdlss-small")
        if job.preview:
            def switch() -> None:
                compare = view.value == "compare"
                player.set_source(f"/api/jobs/{job.id}/preview" if compare else result_url)
                note.set_text("Original on the left, result on the right; the first 12 seconds." if compare else f"{job.input_name} → {job.outputs[0]}")

            view.on_value_change(lambda _e: switch())


@ui.page("/video")
def video_page(job: str | None = None) -> None:
    """``?job=ID`` opens a finished job's side-by-side preview."""
    state = get_state()
    with layout("Video", "Choose a frame, tune the rendering, then export the clip."):
        with ui.element("div").classes("mlxdlss-grid"):
            with ui.element("div").classes("mlxdlss-stack"):
                with ds.card("Source"):
                    zone = ds.dropzone(accept="video/*", title="Drop a video here, or click to browse", hint="MP4, MOV, MKV, WebM · any length",
                                       on_upload=lambda e: on_upload(e), max_size=8_000_000_000)
                    picked = ui.element("div").classes("w-full").style("display: none")
                preview = LivePreview(state, "video")
                results = ui.element("div").classes("mlxdlss-stack")
                opened = state.store.get(job) if job else None
                if opened is not None and opened.kind != "video":
                    opened = None
                if opened is not None and opened.state == "done" and opened.outputs:
                    with results:
                        result_card(opened)
            with ui.element("div").classes("mlxdlss-stack"):
                chain = effect_editor("video", opened.effects if opened is not None else None, on_change=preview.request)
                preview.chain = chain
                output = output_editor(opened.output_options if opened is not None else None)
                ds.button("Export video", kind="primary", icon="play_arrow", large=True, on_click=lambda: submit())
                ds.button("Export all videos", kind="secondary", on_click=lambda: submit(all_files=True)).bind_visibility_from(
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
                ui.notify("Choose a video first.", type="warning"); return
            try:
                files = list(preview.files.items()) if all_files else [(str(preview.source), preview.name)]
                jobs = await run.io_bound(preview.create_jobs, chain(), output(), files)
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
