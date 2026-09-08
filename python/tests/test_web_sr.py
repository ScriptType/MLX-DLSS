"""DLSS SR video routing, persistent settings and native preview/export."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

if (
    importlib.util.find_spec("pydantic") is None
    or importlib.util.find_spec("fastapi") is None
):
    raise unittest.SkipTest("Install the web extra")

from mlxdlss.web.effects import (
    DLSSSuperResolution,
    NeuralRender,
    FrameGen,
    parse_effects,
    validate_chain,
)
from mlxdlss.web.jobs import JobStore
from mlxdlss.web.native import dlss_sr_arguments
from mlxdlss.web.preview import PreviewSession
from mlxdlss.web.runners import JobRunner
from mlxdlss.web.settings import Settings


class DLSSSRWebTests(unittest.TestCase):
    def test_video_chain_rejects_images_duplicates_and_reordering(self):
        sr = parse_effects([{"kind": "sr"}], kind="video")[0]
        self.assertIsInstance(sr, DLSSSuperResolution)
        for effects in (
            [sr],
            [NeuralRender(), sr],
            [FrameGen(), sr],
            [FrameGen(), NeuralRender(), sr],
        ):
            validate_chain(effects, "video")
        for effects, kind in [
            ([sr], "image"),
            ([sr, sr], "video"),
            ([sr, NeuralRender()], "video"),
        ]:
            with self.assertRaises(ValueError):
                validate_chain(effects, kind)
        for values in ({"scale": 4}, {"mode": "other"}):
            with self.assertRaises(ValueError):
                parse_effects([{"kind": "sr", **values}])

    def test_persistence_availability_and_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.srmodel"
            model.mkdir()
            for name in ["manifest.json", "weights.safetensors", "postprocess.metal"]:
                (model / name).write_text("fixture")
            with patch.dict(os.environ, {"MLXDLSS_SR_MODEL": str(model)}):
                settings = Settings.load(Path(directory) / "settings")
            settings.save()
            self.assertEqual(Settings.load(settings.root).sr_model, str(model))
            with (
                patch.object(Settings, "mlxdlss_available", return_value=True),
                patch("mlxdlss.web.native.platform.system", return_value="Darwin"),
                patch(
                    "mlxdlss.web.native.platform.mac_ver", return_value=("26.0", (), "")
                ),
            ):
                self.assertTrue(settings.sr_available())
                settings.backend = "torch"
                self.assertFalse(settings.sr_available())
            self.assertEqual(
                dlss_sr_arguments(DLSSSuperResolution(), settings),
                ["--sr-model", str(model)],
            )
            (model / "manifest.json").unlink()
            with self.assertRaises(ValueError):
                dlss_sr_arguments(DLSSSuperResolution(), settings)

    def test_unsupported_backend_never_drops_sr(self):
        settings = Settings(backend="torch")
        runner = JobRunner(lambda: settings)
        session = PreviewSession(runner)
        self.addCleanup(session.close)
        with self.assertRaisesRegex(ValueError, "DLSS SR needs native Metal"):
            session.render(Path("source.mp4"), "video", [{"kind": "sr"}])
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            job = store.create("source.mp4", [{"kind": "sr"}], data=b"fixture")
            with self.assertRaisesRegex(ValueError, "DLSS SR needs native Metal"):
                runner(job, store.folder(job.id), lambda *_: None, lambda: False)


@unittest.skipUnless(
    os.environ.get("MLXDLSS_NATIVE_PREVIEW_BINARY")
    and os.environ.get("MLXDLSS_SR_MODEL")
    and shutil.which("ffmpeg"),
    "Set native preview binary and SR model; FFmpeg creates the test input",
)
class NativeDLSSSRWebTests(unittest.TestCase):
    def test_preview_api_export_and_optional_frame_generation(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from mlxdlss.web.api import build_router
        from mlxdlss.web.state import WebState
        from mlxdlss.video import probe

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=96x64:rate=10:duration=0.3",
                    "-c:v",
                    "libx264",
                    str(source),
                ],
                check=True,
            )
            settings = Settings(
                root=directory,
                backend="mlxdlss",
                sr_model=os.environ["MLXDLSS_SR_MODEL"],
                mlxdlss_binary=os.environ["MLXDLSS_NATIVE_PREVIEW_BINARY"],
                fg_weights=os.environ.get("MLXDLSS_FG_WEIGHTS", ""),
            )
            state = WebState(settings)
            session = PreviewSession(state.runner)
            app = FastAPI()
            app.include_router(build_router(state))
            try:
                with TestClient(app) as client:
                    capability = next(
                        e
                        for e in client.get("/api/effects").json()["effects"]
                        if e["kind"] == "sr"
                    )
                    self.assertTrue(capability["available"])
                    first = session.render(source, "video", [{"kind": "sr"}], 0.2)
                    self.assertEqual((first["width"], first["height"]), (192, 128))
                    self.assertEqual(first["historyFrames"], 2)
                    self.assertEqual(
                        session.render(source, "video", [{"kind": "sr"}], 0.2)[
                            "processed"
                        ],
                        first["processed"],
                    )
                    bypass = session.render(source, "video", [], 0.2)
                    self.assertEqual(bypass["processed"], bypass["original"])
                    chains = [[{"kind": "sr"}]]
                    if settings.fg_weights:
                        chains.append([{"kind": "fg", "factor": 2}, {"kind": "sr"}])
                    for effects in chains:
                        response = client.post(
                            "/api/jobs",
                            files={
                                "file": (source.name, source.read_bytes(), "video/mp4")
                            },
                            data={"effects": json.dumps(effects)},
                        )
                        self.assertEqual(response.status_code, 201, response.text)
                        job_id = response.json()["id"]
                        deadline = time.monotonic() + 60
                        while time.monotonic() < deadline:
                            job = client.get(f"/api/jobs/{job_id}").json()
                            if job["state"] not in {"queued", "running"}:
                                break
                            time.sleep(0.05)
                        self.assertEqual(job["state"], "done", job.get("error"))
                        self.assertEqual(job["diagnostics"]["pipeline"], "native")
                        self.assertTrue(job["diagnostics"]["temporal"])
                        info = probe(state.store.folder(job_id) / job["outputs"][0])
                        self.assertEqual((info.width, info.height), (192, 128))
                        self.assertEqual(
                            info.frame_count, 3 if len(effects) == 1 else 5
                        )
            finally:
                session.close()
                state.close()


if __name__ == "__main__":
    unittest.main()
