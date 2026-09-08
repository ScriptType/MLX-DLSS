"""Page frame (header, navigation, theme) and the effect editor shared by the Image and Video pages."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from nicegui import ui

from ..effects import PROFILE_NAMES
from ..state import get_state
from . import ds

NAV = [("Image", "/"), ("Video", "/video"), ("Jobs", "/jobs"), ("Settings", "/settings")]


@contextmanager
def layout(title: str, lead: str | None = None):
    state = get_state()
    settings = state.settings
    ui.page_title(f"{title} · MLX-DLSS")
    ui.colors(primary="#4cc2f0" if settings.theme_dark else "#0b6e99", positive="#1f8a4c", negative="#c0392b", warning="#b7791f")
    ds.install()
    dark = ui.dark_mode()
    if settings.theme_dark:
        dark.enable()
    else:
        dark.disable()

    def toggle_theme() -> None:
        dark.toggle()
        state.update_settings(theme_dark=bool(dark.value))
        theme_button.props(f"icon={'light_mode' if dark.value else 'dark_mode'}")

    with ui.header().classes("mlxdlss-header"):
        with ui.element("div").classes("mlxdlss-header-inner"):
            with ui.element("div").classes("flex items-center"):
                with ui.element("a").classes("mlxdlss-brand").props("href=/"):
                    ui.element("span").classes("mlxdlss-glyph").props("innerHTML=N")
                    ui.label("MLX-DLSS")
                with ui.element("nav").classes("mlxdlss-nav"):
                    for name, path in NAV:
                        ui.link(name, path).classes("active" if name == title else "")
            with ui.element("div").classes("mlxdlss-status"):
                queue_label = ui.label()
                backend = settings.backend + (" · Metal" if settings.mlxdlss_available() else "")
                ui.label(backend)
                theme_button = ui.button(icon="light_mode" if settings.theme_dark else "dark_mode", on_click=toggle_theme).props("flat round dense").classes("mlxdlss-iconbtn")
                theme_button.tooltip("Switch theme")
        rail = ui.element("div").classes("mlxdlss-rail")

    def refresh_status() -> None:
        active = sum(1 for j in state.store.list() if j.state in ("queued", "running"))
        queue_label.set_text(f"{active} running" if active else "idle")
        rail.classes(add="active" if active else "", remove="" if active else "active")

    refresh_status()
    ui.timer(1.0, refresh_status)
    with ui.element("main").classes("mlxdlss-page"):
        if lead is not None:
            ds.page_head(title, lead)
        yield


def effect_editor(kind: str, initial: list[dict] | None = None, *, on_change=lambda: None) -> Callable[[], list[dict]]:
    """Effect cards for ``kind`` (image | video); returns a getter for the ordered chain.

    ``initial`` (a job's effect list) preloads the controls, so a reopened result shows what produced it."""
    settings = get_state().settings
    initial = initial or []
    nr0 = next((e for e in initial if e.get("kind") == "nr"), None)
    fg0 = next((e for e in initial if e.get("kind") == "fg"), None)
    with ds.card("Neural rendering") as box:
        with box.meta:
            nr_enabled = ui.switch(value=nr0 is not None or not initial).props('dense color=primary aria-label="Enable rendering"')
        if not settings.has_nr_weights():
            ui.label("Weights are not configured yet — add them in Settings.").classes("mlxdlss-warn mlxdlss-small")
        box.body.bind_visibility_from(nr_enabled, "value")
        with ui.element("div").classes("mlxdlss-stack"):
            nr0 = nr0 or {}
            profile = ds.segmented_row("Profile", list(PROFILE_NAMES), value=nr0.get("profile", "standard"))
            scale = ds.slider_row("Processing scale", value=float(nr0.get("processing_scale", 1.0)), minimum=1.0, maximum=4.0, step=0.1, hint="Higher values add detail and processing time.")
            detail = ds.slider_row("Detail", value=float(nr0.get("detail_strength", 1.0)), minimum=0.0, maximum=8.0, step=0.1)
            colour = ds.slider_row("Colour", value=float(nr0.get("colour_strength", 1.0)), minimum=0.0, maximum=4.0, step=0.1)
            with ui.expansion("Detail radius").classes("w-full"):
                radius = ds.slider_row("Radius", value=float(nr0.get("detail_radius", 4.0)), minimum=0.5, maximum=64.0, step=0.5)
            intensity = ds.slider_row("Intensity", value=float(nr0.get("intensity", 1.0)), minimum=0.0, maximum=2.0, step=0.1)
            temporal = motion = threshold = None
            if kind == "video":
                temporal = ds.switch_row("Temporal", "Keep generated detail stable using motion and reliable history.", value=bool(nr0.get("temporal", not initial)))
                with ui.expansion("Motion and scene cuts").classes("w-full"):
                    motion = ui.select({"automatic": "Automatic", "videotoolbox": "VideoToolbox", "vision": "Vision",
                                        "flow": "OpenCV", "zero": "Zero (diagnostic)"}, value=nr0.get("motion", "automatic"), label="Motion").props("outlined dense").classes("w-full")
                    ui.label("VideoToolbox and Vision use native Metal media on macOS 26.").classes("mlxdlss-muted mlxdlss-small")
                    threshold = ds.slider_row("Scene cut threshold", value=nr0.get("scene_cut_threshold", 0.3), minimum=0, maximum=1, step=0.01)
    fg_enabled = fg_mode = fg_factor = fg_audio = order = None
    if kind == "video":
        with ds.card("Frame generation") as box:
            with box.meta:
                fg_enabled = ui.switch(value=fg0 is not None).props('dense color=primary aria-label="Generate frames"')
            if not settings.has_fg_weights():
                ui.label("Weights are not configured yet — run mlxdlss-weights extract-fg and add the file in Settings.").classes("mlxdlss-warn mlxdlss-small")
            box.body.bind_visibility_from(fg_enabled, "value")
            with ui.element("div").classes("mlxdlss-stack"):
                fg0 = fg0 or {}
                fg_mode = ds.segmented_row("Mode", {"fps": "Higher frame rate", "slowmo": "Slow motion"}, value=fg0.get("mode", "fps"))
                fg_factor = ds.segmented_row("Factor", {2: "×2", 3: "×3", 4: "×4", 8: "×8", 16: "×16"}, value=int(fg0.get("factor", 2)))
                fg_audio = ds.segmented_row("Audio", {"copy": "Copy", "stretch": "Stretch", "none": "Drop"}, value=fg0.get("audio", "copy"),
                                            hint="Stretch keeps the pitch and only applies to slow motion.")
                fg_mode.on_value_change(lambda e: fg_audio.set_value("stretch" if e.value == "slowmo" else "copy"))
                first = initial[0].get("kind") if initial else "nr"
                order = ds.segmented_row("Order", {"nr_first": "Render → generate", "fg_first": "Generate → render"},
                                         value="fg_first" if first == "fg" else "nr_first")

    def chain() -> list[dict]:
        effects: list[dict] = []
        if nr_enabled.value:
            nr = {"kind": "nr", "profile": profile.value, "processing_scale": float(scale.value), "detail_strength": float(detail.value),
                  "colour_strength": float(colour.value), "detail_radius": float(radius.value), "intensity": float(intensity.value)}
            if temporal is not None:
                nr["temporal"] = bool(temporal.value)
                nr["motion"] = motion.value
                nr["scene_cut_threshold"] = float(threshold.value)
            effects.append(nr)
        if fg_enabled is not None and fg_enabled.value:
            fg = {"kind": "fg", "mode": fg_mode.value, "factor": int(fg_factor.value), "audio": fg_audio.value}
            if order is not None and order.value == "fg_first":
                effects.insert(0, fg)
            else:
                effects.append(fg)
        return effects

    for control in (nr_enabled, profile, scale, detail, colour, radius, intensity, temporal, motion, threshold,
                    fg_enabled, fg_mode, fg_factor, fg_audio, order):
        if control is not None:
            control.on_value_change(lambda _: on_change())
    return chain


def output_editor(initial: dict | None = None):
    values = initial or {}
    with ds.card("Output"):
        codec = ui.select({"h264": "H.264", "hevc": "HEVC", "prores": "ProRes 422 HQ"},
                          value=values.get("codec", "h264"), label="Video codec").props("outlined dense").classes("w-full")
        audio = ui.switch("Include audio", value=values.get("include_audio", True))
        with ui.expansion("Video range").classes("w-full"):
            start = ui.number("Start frame", value=values.get("start_frame", 0), min=0, step=1, precision=0).props("outlined dense")
            limit = ui.number("Frame limit (0 = all)", value=values.get("frame_limit") or 0, min=0, step=1, precision=0).props("outlined dense")
        ui.link("Output folder in Settings", "/settings").classes("mlxdlss-small")
    return lambda: {"codec": codec.value, "include_audio": bool(audio.value),
                    "start_frame": int(start.value or 0), "frame_limit": int(limit.value) if limit.value else None}


def job_status_card(job_id: str, *, on_done: Callable[[], None] | None = None) -> None:
    """A live card following one job until it finishes."""
    state = get_state()
    with ds.card("Job") as box:
        with box.meta:
            status = ds.chip("queued")
        name = ui.label().classes("font-medium")
        stage = ui.label().classes("mlxdlss-muted mlxdlss-small")
        bar = ds.progress()
        with ui.element("div").classes("flex items-center gap-2"):
            cancel = ds.button("Cancel", kind="danger", on_click=lambda: state.queue.cancel(job_id))
            ds.button("All jobs", kind="ghost", on_click=lambda: ui.navigate.to("/jobs"))

    def refresh() -> None:
        job = state.store.get(job_id)
        if job is None:
            timer.deactivate(); return
        name.set_text(job.input_name)
        status.set_text(job.state); status.classes(replace=f"mlxdlss-chip mlxdlss-chip-{job.state}")
        stage.set_text((job.stage or "waiting for the queue") + (f" · {job.seconds:.0f} s" if job.seconds else ""))
        bar.value = job.progress
        if job.state in ("done", "failed", "cancelled"):
            cancel.disable(); timer.deactivate()
            if job.state == "failed":
                stage.set_text(job.error or "failed"); stage.classes(add="mlxdlss-bad")
            if on_done is not None:
                on_done()

    timer = ui.timer(0.5, refresh)
