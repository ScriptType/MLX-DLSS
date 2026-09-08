"""RTX VSR image routing and real Metal preview/export parity."""
import base64
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

if importlib.util.find_spec("pydantic") is None or importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest("Install the web extra")

from mlxdlss.web.effects import NeuralRender, SuperResolution, parse_effects, validate_chain
from mlxdlss.web.preview import PreviewSession
from mlxdlss.web.runners import JobRunner
from mlxdlss.web.settings import Settings


class SuperResolutionWebTests(unittest.TestCase):
    def test_image_chain_and_unsupported_combinations(self):
        vsr = parse_effects([{"kind": "vsr"}], kind="image")[0]
        self.assertIsInstance(vsr, SuperResolution)
        self.assertEqual(vsr.scale, 2)
        validate_chain([vsr], "image")
        validate_chain([NeuralRender(), vsr], "image")
        for chain, kind in (([vsr], "video"), ([vsr, vsr], "image"), ([vsr, NeuralRender()], "image")):
            with self.subTest(chain=chain, kind=kind), self.assertRaises(ValueError):
                validate_chain(chain, kind)
        for options in ({"scale": 4}, {"mode": 32}):
            with self.assertRaises(ValueError):
                parse_effects([{"kind": "vsr", **options}])

    def test_settings_persist_weights_and_report_only_supported_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "vsr.safetensors"
            weights.write_bytes(b"fixture")
            with patch.dict(os.environ, {"MLXDLSS_VSR_WEIGHTS": str(weights)}):
                settings = Settings.load(directory)
            self.assertEqual(settings.vsr_weights, str(weights))
            settings.save()
            self.assertEqual(Settings.load(directory).vsr_weights, str(weights))
            with patch.object(Settings, "mlxdlss_available", return_value=True), \
                 patch("mlxdlss.web.native.platform.system", return_value="Darwin"), \
                 patch("mlxdlss.web.native.platform.mac_ver", return_value=("26.0", (), "")):
                self.assertTrue(settings.vsr_available())
                settings.backend = "torch"
                self.assertFalse(settings.vsr_available())
                settings.backend = "mlxdlss"
                with patch("mlxdlss.web.native.platform.mac_ver", return_value=("15.0", (), "")):
                    self.assertFalse(settings.vsr_available())
                settings.vsr_weights = str(weights.with_name("missing"))
                self.assertFalse(settings.vsr_available())

    def test_preview_and_export_never_silently_drop_vsr(self):
        settings = Settings(backend="torch")
        runner = JobRunner(lambda: settings)
        session = PreviewSession(runner)
        self.addCleanup(session.close)
        source = Path("source.png")
        with self.assertRaisesRegex(ValueError, "RTX VSR needs native Metal"):
            session.render(source, "image", [{"kind": "vsr"}])
        with self.assertRaisesRegex(ValueError, "RTX VSR needs native Metal"):
            runner._image(source, Path("."), [SuperResolution()], settings, lambda *_: None)


@unittest.skipUnless(os.environ.get("MLXDLSS_NATIVE_PREVIEW_BINARY") and os.environ.get("MLXDLSS_VSR_WEIGHTS"),
                     "Set MLXDLSS_NATIVE_PREVIEW_BINARY and MLXDLSS_VSR_WEIGHTS")
class NativeSuperResolutionWebTests(unittest.TestCase):
    def test_preview_toggle_error_recovery_and_api_export_match(self):
        self._roundtrip([{"kind": "vsr"}])

    @unittest.skipUnless(os.environ.get("MLXDLSS_NR_MODEL"), "Set MLXDLSS_NR_MODEL")
    def test_render_then_upscale_uses_the_same_preview_and_export_chain(self):
        self._roundtrip([{"kind": "nr"}, {"kind": "vsr"}])

    def _roundtrip(self, effects):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from mlxdlss.web.api import build_router
        from mlxdlss.web.state import WebState

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            pixels = np.random.default_rng(42).integers(0, 256, (65, 97, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(source)
            settings = Settings(root=directory, backend="mlxdlss", vsr_weights=os.environ["MLXDLSS_VSR_WEIGHTS"],
                                mlxdlss_binary=os.environ["MLXDLSS_NATIVE_PREVIEW_BINARY"], nr_model=os.environ.get("MLXDLSS_NR_MODEL", ""))
            state = WebState(settings)
            session = PreviewSession(state.runner)
            app = FastAPI()
            app.include_router(build_router(state))
            try:
                with TestClient(app) as client:
                    capability = next(e for e in client.get("/api/effects").json()["effects"] if e["kind"] == "vsr")
                    self.assertTrue(capability["available"])
                    self.assertEqual(capability["media"], ["image"])
                    first = session.render(source, "image", effects)
                    child = session.process
                    self.assertEqual((first["width"], first["height"]), (194, 130))
                    with Image.open(io.BytesIO(base64.b64decode(first["processed"]))) as image:
                        preview = np.array(image.convert("RGB"))
                    with Image.open(io.BytesIO(base64.b64decode(first["original"]))) as image:
                        self.assertEqual(image.size, (97, 65))
                    bypass = session.render(source, "image", [])
                    self.assertEqual(bypass["processed"], bypass["original"])
                    self.assertEqual((bypass["width"], bypass["height"]), (97, 65))
                    weights = settings.vsr_weights
                    bad = Path(directory) / "bad.safetensors"
                    bad.write_bytes(b"invalid")
                    settings.vsr_weights = str(bad)
                    with self.assertRaises(ValueError):
                        session.render(source, "image", effects)
                    settings.vsr_weights = weights
                    self.assertEqual(session.render(source, "image", effects)["processed"], first["processed"])
                    self.assertIs(session.process, child, "Settings changes must reuse the preview process")
                    import json
                    response = client.post("/api/jobs", files={"file": (source.name, source.read_bytes(), "image/png")},
                                           data={"effects": json.dumps(effects)})
                    self.assertEqual(response.status_code, 201, response.text)
                    job_id = response.json()["id"]
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        job = client.get(f"/api/jobs/{job_id}").json()
                        if job["state"] not in {"queued", "running"}:
                            break
                        time.sleep(0.05)
                    self.assertEqual(job["state"], "done", job.get("error"))
                    result = client.get(f"/api/jobs/{job_id}/download/0")
                    self.assertEqual(result.status_code, 200)
                    with Image.open(io.BytesIO(result.content)) as image:
                        self.assertEqual(image.size, (194, 130))
                        np.testing.assert_array_equal(np.array(image.convert("RGB")), preview)
            finally:
                session.close()
                state.close()
