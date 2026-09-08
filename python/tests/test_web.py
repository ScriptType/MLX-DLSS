import io
import json
import pathlib
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import importlib.util

import numpy as np

if importlib.util.find_spec("pydantic") is None or importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest("the web front end needs the 'web' extra (pip install 'mlxdlss[web]')")

from mlxdlss.web.effects import FrameGen, NeuralRender, describe_effects, media_kind, parse_effects, validate_chain
from mlxdlss.web.jobs import JobQueue, JobStore
from mlxdlss.web.settings import Settings

from .synthetic import synthetic_weights, write_logical_safetensors
from .test_framegen import synthetic_framegen_weights

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class EffectTests(unittest.TestCase):
    def test_native_control_ranges_and_motion_are_preserved(self):
        nr = parse_effects([{"kind": "nr", "detail_strength": 8, "detail_radius": 0.5,
                             "motion": "vision", "scene_cut_threshold": 0.15}])[0]
        self.assertEqual(nr.model_dump()["motion"], "vision")
        self.assertEqual(nr.model_dump()["scene_cut_threshold"], 0.15)
        self.assertEqual(FrameGen(factor=16).factor, 16)
        for change in ({"detail_strength": 9}, {"detail_radius": 65}, {"motion": "invalid"},
                       {"scene_cut_threshold": float("nan")}):
            with self.assertRaises(ValueError):
                parse_effects([{"kind": "nr", **change}])

    def test_media_kind(self):
        self.assertEqual(media_kind("clip.MP4"), "video")
        self.assertEqual(media_kind("photo.jpeg"), "image")
        with self.assertRaises(ValueError):
            media_kind("notes.txt")

    def test_parse_and_validate(self):
        effects = parse_effects([{"kind": "nr", "profile": "standard"}, {"kind": "fg", "mode": "slowmo", "factor": 4, "audio": "stretch"}])
        self.assertIsInstance(effects[0], NeuralRender)
        self.assertIsInstance(effects[1], FrameGen)
        validate_chain(effects, "video")
        with self.assertRaises(ValueError):
            validate_chain(effects, "image")
        with self.assertRaises(ValueError):
            validate_chain([], "video")
        with self.assertRaises(ValueError):
            validate_chain([effects[1], FrameGen()], "video")
        with self.assertRaises(ValueError):
            parse_effects([{"kind": "fg", "mode": "fps", "audio": "stretch"}])
        with self.assertRaises(ValueError):
            parse_effects([{"kind": "nr", "profile": "no-such-profile"}])
        self.assertEqual(parse_effects([{"kind": "nr", "temporal": True, "processing_scale": 2}])[0].processing_scale, 2)
        with self.assertRaises(ValueError):
            parse_effects([{"kind": "fg", "factor": 5}])

    def test_describe(self):
        d = describe_effects(mlxdlss_available=False, fg_weights=True, nr_weights=False)
        kinds = {e["kind"]: e for e in d["effects"]}
        self.assertTrue(kinds["fg"]["available"]); self.assertFalse(kinds["nr"]["available"])
        self.assertEqual(kinds["fg"]["fields"]["factor"]["choices"], [2, 3, 4, 8, 16])


class JobQueueTests(unittest.TestCase):
    def test_clear_finished_hides_history_without_deleting_results(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(pathlib.Path(directory))
            done = store.create("done.png", [{"kind": "nr"}], data=b"original")
            queued = store.create("queued.png", [{"kind": "nr"}], data=b"queued")
            done.state = "done"
            store.clear_finished()
            self.assertEqual([j.id for j in JobStore(store.root).list()], [queued.id])
            self.assertEqual(store.input_path(done).read_bytes(), b"original")

    def test_output_folder_switch_keeps_existing_results_and_rejects_active_jobs(self):
        from mlxdlss.web.state import WebState
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            state = WebState(Settings(root=str(root / "original")), runner=lambda *_: [])
            job = state.store.create("a.png", [{"kind": "nr"}], data=b"source")
            with self.assertRaises(ValueError):
                state.update_settings(output_directory=str(root / "new"))
            job.state = "done"
            state.store.save(job)
            original = state.store.input_path(job)
            state.update_settings(output_directory=str(root / "new"))
            self.assertEqual(state.store.root, root / "new")
            self.assertEqual(original.read_bytes(), b"source")
            self.assertEqual(Settings.load(root / "original").outputs, root / "new")
            state.close()

    def test_new_video_default_and_saved_opt_out_survive_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            store = JobStore(root)
            new = store.create("new.mp4", [{"kind": "nr"}], data=b"data")
            old = store.create("old.mp4", [{"kind": "nr", "temporal": False}], data=b"data")
            image = store.create("photo.png", [{"kind": "nr"}], data=b"data")
            again = JobStore(root)
            self.assertTrue(again.get(new.id).effects[0]["temporal"])
            self.assertFalse(again.get(old.id).effects[0]["temporal"])
            self.assertFalse(again.get(image.id).effects[0]["temporal"])

    def test_store_queue_cancel_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            store = JobStore(root)
            seen = []

            def runner(job, folder, report, should_stop):
                for i in range(5):
                    if should_stop():
                        return []
                    report("step", i / 5, i, 5); time.sleep(0.05)
                out = folder / "result.png"; out.write_bytes(b"x")
                return [out]

            queue = JobQueue(store, runner, on_change=lambda j: seen.append(j.state))
            job = store.create("a.png", [{"kind": "nr"}], data=b"data")
            self.assertTrue((store.folder(job.id) / "input.png").exists())
            queue.submit(job)
            for _ in range(100):
                if store.get(job.id).state == "done":
                    break
                time.sleep(0.05)
            self.assertEqual(store.get(job.id).state, "done")
            self.assertEqual(store.get(job.id).outputs, ["result.png"])
            self.assertIn("running", seen)
            # cancellation of a running job
            job2 = store.create("b.png", [{"kind": "nr"}], data=b"data"); queue.submit(job2)
            time.sleep(0.08); queue.cancel(job2.id)
            for _ in range(100):
                if store.get(job2.id).state in ("cancelled", "done"):
                    break
                time.sleep(0.05)
            self.assertEqual(store.get(job2.id).state, "cancelled")
            # invalid chain is rejected before a folder exists
            with self.assertRaises(ValueError):
                store.create("c.png", [{"kind": "fg"}], data=b"x")
            # a fresh store reloads the finished jobs from disk
            again = JobStore(root)
            self.assertEqual({j.id for j in again.list()}, {job.id, job2.id})

    def test_runner_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(pathlib.Path(directory))

            def runner(job, folder, report, should_stop):
                raise RuntimeError("boom")

            queue = JobQueue(store, runner)
            job = store.create("a.png", [{"kind": "nr"}], data=b"data"); queue.submit(job)
            for _ in range(100):
                if store.get(job.id).state == "failed":
                    break
                time.sleep(0.05)
            self.assertEqual(store.get(job.id).state, "failed")
            self.assertIn("boom", store.get(job.id).error)
            self.assertTrue((store.folder(job.id) / "error.log").exists())


def _png_bytes(width: int = 48, height: int = 32) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    Image.fromarray((rng.random((height, width, 3)) * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


class ApiAndRunnerTests(unittest.TestCase):
    """The FastAPI routes on a WebState with synthetic weights (no NiceGUI server)."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        root = pathlib.Path(cls.directory)
        cls.nr_weights = root / "nr.safetensors"
        write_logical_safetensors(cls.nr_weights, synthetic_weights())
        cls.fg_weights = root / "fg.safetensors"
        from safetensors.torch import save_file

        save_file(synthetic_framegen_weights(), str(cls.fg_weights), metadata={"format": "dlssg-framegen-dense-v1"})
        settings = Settings(root=str(root / "web"), nr_weights=str(cls.nr_weights), fg_weights=str(cls.fg_weights), backend="torch", device="cpu")
        from mlxdlss.web.state import WebState
        from mlxdlss.web.api import build_router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        cls.state = WebState(settings)
        app = FastAPI()
        app.include_router(build_router(cls.state))
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.state.close()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def _wait(self, job_id: str, timeout: float = 120.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.client.get(f"/api/jobs/{job_id}").json()
            if data["state"] in ("done", "failed", "cancelled"):
                return data
            time.sleep(0.1)
        self.fail("job did not finish")

    def test_effects_endpoint(self):
        data = self.client.get("/api/effects").json()
        self.assertEqual({e["kind"] for e in data["effects"]}, {"nr", "fg", "vsr", "sr"})
        self.assertTrue(data["backends"]["torch"])

    def test_image_job_runs_neural_rendering(self):
        response = self.client.post("/api/jobs", files={"file": ("photo.png", _png_bytes(), "image/png")},
                                    data={"effects": json.dumps([{"kind": "nr", "profile": "standard"}])})
        self.assertEqual(response.status_code, 201, response.text)
        job = self._wait(response.json()["id"])
        self.assertEqual(job["state"], "done", job.get("error"))
        self.assertEqual(job["outputs"], ["result.png"])
        download = self.client.get(f"/api/jobs/{job['id']}/download/0")
        self.assertEqual(download.status_code, 200)
        self.assertTrue(download.content.startswith(b"\x89PNG"))
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/input").status_code, 200)

    def test_image_export_honors_exif_orientation(self):
        from PIL import Image
        buffer = io.BytesIO()
        image = Image.new("RGB", (32, 24), (60, 90, 120))
        exif = image.getexif()
        exif[274] = 6
        image.save(buffer, format="JPEG", exif=exif)
        response = self.client.post("/api/jobs", files={"file": ("rotated.jpg", buffer.getvalue(), "image/jpeg")},
                                    data={"effects": json.dumps([{"kind": "nr", "intensity": 0}])})
        self.assertEqual(response.status_code, 201)
        job = self._wait(response.json()["id"])
        self.assertEqual(job["state"], "done", job.get("error"))
        download = self.client.get(f"/api/jobs/{job['id']}/download/0")
        with Image.open(io.BytesIO(download.content)) as output:
            self.assertEqual(output.size, (24, 32))

    def test_metal_image_keeps_older_macos_command(self):
        from mlxdlss.web.runners import JobRunner
        settings = Settings(backend="mlxdlss", nr_model="model.dlssmodel")
        runner = JobRunner(lambda: settings)
        with patch("mlxdlss.web.native.platform.mac_ver", return_value=("15.0", (), "")), \
             patch("mlxdlss.web.native.platform.system", return_value="Darwin"), \
             patch("mlxdlss.mlxdlss_stream.find_mlxdlss", return_value="mlxdlss"), \
             patch("mlxdlss.web.native.run_media") as run:
            runner._image(pathlib.Path("source.png"), pathlib.Path(self.directory), [NeuralRender()], settings, lambda *_: None)
            command = run.call_args.args[0]
            self.assertEqual(command[1:4], ["render-image", "source.png", "model.dlssmodel"])
            self.assertIn("--intensity", command)
            self.assertNotIn("--model", command)

    def test_bad_requests(self):
        for entry in (None, 42, False):
            with self.subTest(entry=entry):
                response = self.client.post("/api/jobs", files={"file": ("clip.mp4", b"invalid", "video/mp4")},
                                            data={"effects": json.dumps([entry])})
                self.assertEqual(response.status_code, 400, response.text)
        response = self.client.post("/api/jobs", files={"file": ("photo.png", _png_bytes(), "image/png")},
                                    data={"effects": json.dumps([{"kind": "fg"}])})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/jobs/nope").status_code, 404)
        self.assertEqual(self.client.post("/api/jobs/nope/cancel").status_code, 404)

    def test_output_options_are_validated_and_persisted(self):
        options = {"codec": "prores", "include_audio": False, "start_frame": 2, "frame_limit": 3}
        response = self.client.post("/api/jobs", files={"file": ("clip.mp4", b"invalid", "video/mp4")},
                                    data={"effects": '[{"kind":"nr"}]', "output_options": json.dumps(options)})
        self.assertEqual(response.status_code, 201, response.text)
        job = self._wait(response.json()["id"])
        self.assertEqual(job.get("output_options"), options)
        self.assertEqual(JobStore(self.state.settings.outputs).get(job["id"]).output_options, options)
        for invalid in ({"frame_limit": 0}, {"start_frame": -1}, {"codec": "wrong"}):
            response = self.client.post("/api/jobs", files={"file": ("clip.mp4", b"invalid", "video/mp4")},
                                        data={"effects": '[{"kind":"nr"}]', "output_options": json.dumps(invalid)})
            self.assertEqual(response.status_code, 400, response.text)

    def test_retry_clones_failed_job_with_its_output_settings(self):
        job = self.state.store.create("clip.mp4", [{"kind": "nr"}], data=b"invalid",
                                      output_options={"frame_limit": 2})
        self.assertEqual(self.client.post(f"/api/jobs/{job.id}/retry").status_code, 409)
        job.state = "failed"
        self.state.store.save(job)
        response = self.client.post(f"/api/jobs/{job.id}/retry")
        self.assertEqual(response.status_code, 201, response.text)
        retry = self._wait(response.json()["id"])
        self.assertNotEqual(retry["id"], job.id)
        self.assertEqual(retry["output_options"]["frame_limit"], 2)
        self.assertEqual(self.state.store.input_path(job).read_bytes(), b"invalid")

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
    def test_trim_codec_and_audio_apply_to_the_complete_chain(self):
        from mlxdlss.video import probe
        src = pathlib.Path(self.directory) / "range.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=0.5",
                        "-f", "lavfi", "-i", "sine=duration=0.5", "-c:v", "libx264", "-c:a", "aac", str(src)], check=True)
        response = self.client.post("/api/jobs", files={"file": ("range.mp4", src.read_bytes(), "video/mp4")},
            data={"effects": json.dumps([{"kind": "nr"}, {"kind": "fg", "factor": 4, "mode": "slowmo", "audio": "stretch"}]),
                  "output_options": json.dumps({"codec": "prores", "start_frame": 1, "frame_limit": 2, "include_audio": False})})
        self.assertEqual(response.status_code, 201, response.text)
        job = self._wait(response.json()["id"])
        self.assertEqual(job["state"], "done", job.get("error"))
        info = probe(self.state.store.folder(job["id"]) / "result.mov")
        self.assertEqual(info.frame_count, 5)
        self.assertEqual(info.fps, 10)
        self.assertEqual(info.codec, "prores")
        self.assertFalse(info.has_audio)
        self.assertIsNone(job["preview"], "A trimmed slowmo clip must not show a falsely aligned comparison")

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available")
    def test_video_job_chains_frame_generation_after_neural_rendering(self):
        from mlxdlss.video_pipeline import NeuralRenderStage

        src = pathlib.Path(self.directory) / "src.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
        for effects in ([{"kind": "nr"}, {"kind": "fg", "factor": 2}], [{"kind": "fg", "factor": 2}, {"kind": "nr"}]):
            commands = []
            nr_inputs = []
            popen = subprocess.Popen
            apply = NeuralRenderStage.apply

            def tracked(command, *args, **kwargs):
                commands.append(command)
                return popen(command, *args, **kwargs)

            def observe_nr_inputs(stage, frames, width, height, **kwargs):
                def observed_frames():
                    try:
                        for frame in frames:
                            nr_inputs.append(np.array(frame, copy=True))
                            yield frame
                    finally:
                        close = getattr(frames, "close", None)
                        if close is not None:
                            close()

                return apply(stage, observed_frames(), width, height, **kwargs)

            with patch("subprocess.Popen", side_effect=tracked), patch.object(NeuralRenderStage, "apply", new=observe_nr_inputs):
                response = self.client.post("/api/jobs", files={"file": ("clip.mp4", src.read_bytes(), "video/mp4")},
                                            data={"effects": json.dumps(effects)})
                self.assertEqual(response.status_code, 201, response.text)
                job = self._wait(response.json()["id"], timeout=600)
            self.assertEqual(job["state"], "done", job.get("error"))
            self.assertEqual(job["outputs"], ["result.mp4"])
            from mlxdlss.video import probe

            info = probe(self.state.store.folder(job["id"]) / "result.mp4")
            self.assertEqual(info.frame_count, 7)   # 4 frames -> 4 + 3 generated
            self.assertAlmostEqual(info.fps, 20.0, places=3)
            self.assertEqual(job["preview"], "preview.mp4")
            self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/preview").status_code, 200)
            decoders = [c for c in commands if pathlib.Path(c[0]).name == "ffmpeg" and c[-1] == "-"]
            self.assertEqual(len(decoders), 1, "the effect chain must decode once without intermediate video files")
            if effects[0]["kind"] == "fg":
                self.assertEqual(len(nr_inputs), 7)
                generated = nr_inputs[1::2]
                self.assertTrue(all(np.any(np.abs(frame * 255 - np.rint(frame * 255)) > 1e-5) for frame in generated),
                                "generated frames must retain fractional 8-bit detail until neural rendering")
                self.assertTrue(all(frame.dtype == np.float32 for frame in generated))
