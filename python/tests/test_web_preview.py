"""Live preview contracts; native protocol checks are opt-in on macOS 26."""
import base64
import asyncio
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import tempfile
import unittest
import threading
import importlib.util

from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("pydantic"), "Install the web extra")
class NativeProcessTests(unittest.TestCase):
    def test_cancel_reaps_child_and_discards_private_partial_files(self):
        from mlxdlss.web.native import run_media
        from mlxdlss.web.runners import Cancelled
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.mp4"
            script = "import os,sys,time; from pathlib import Path; Path(sys.argv[2]).write_bytes(b'partial'); Path(os.environ['TMPDIR'],'audio.wav').write_bytes(b'audio'); time.sleep(20)"
            started = time.monotonic()
            with self.assertRaises(Cancelled):
                run_media([sys.executable, "-c", script, "--output", str(target)], lambda *_: None,
                          lambda: time.monotonic() - started > 0.3)
            self.assertLess(time.monotonic() - started, 3)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_publication_preserves_existing_output(self):
        from mlxdlss.web.native import run_media
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.png"
            script = "import json,sys; from pathlib import Path; Path(sys.argv[2]).write_bytes(b'rendered'); print(json.dumps({'output':sys.argv[2]}))"
            command = [sys.executable, "-c", script, "--output", str(target)]
            run_media(command, lambda *_: None, lambda: False)
            self.assertEqual(target.read_bytes(), b"rendered")
            target.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                run_media(command, lambda *_: None, lambda: False)
            self.assertEqual(target.read_bytes(), b"keep")


@unittest.skipUnless(importlib.util.find_spec("pydantic"), "Install the web extra")
class LatestPreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_changes_during_inference_publish_only_the_latest_result(self):
        from mlxdlss.web.preview import LatestPreview
        entered, release = threading.Event(), threading.Event()
        calls, results, errors = [], [], []

        def render(value):
            calls.append(value)
            if value == 1:
                entered.set()
                release.wait(3)
            return value

        worker = LatestPreview(render, results.append, errors.append, delay=0)
        worker.submit(1)
        await asyncio.to_thread(entered.wait, 3)
        worker.submit(2)
        worker.submit(3)
        release.set()
        await worker.task
        self.assertEqual(calls, [1, 3])
        self.assertEqual(results, [3])
        self.assertEqual(errors, [])

    async def test_error_recovers_and_closed_page_discards_pending_work(self):
        from mlxdlss.web.preview import LatestPreview
        results, errors = [], []

        def render(value):
            if value == 1:
                raise ValueError("bad model")
            return value

        worker = LatestPreview(render, results.append, errors.append, delay=0)
        worker.submit(1)
        await worker.task
        worker.submit(2)
        await worker.task
        worker.submit(3)
        worker.close()
        await worker.task
        self.assertEqual(results, [2])
        self.assertEqual(errors, ["bad model"])


@unittest.skipUnless(os.environ.get("MLXDLSS_NATIVE_PREVIEW_BINARY"), "Set MLXDLSS_NATIVE_PREVIEW_BINARY")
class NativePreviewProtocolTests(unittest.TestCase):
    def test_bypass_error_and_recovery_in_one_process(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (64, 48), (30, 90, 160)).save(source)
            request = {"input": str(source), "video": False, "time": 0, "options": []}
            invalid = {**request, "options": ["--model", str(Path(directory) / "missing.dlssmodel")]}
            result = subprocess.run([os.environ["MLXDLSS_NATIVE_PREVIEW_BINARY"], "preview-stream"],
                input="\n".join(map(json.dumps, [request, invalid, request])) + "\n",
                text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            first, error, again = map(json.loads, result.stdout.splitlines())
            self.assertIn("error", error)
            self.assertEqual(first["original"], first["processed"])
            self.assertEqual(first["processed"], again["processed"])
            image = Image.open(io.BytesIO(base64.b64decode(first["processed"])))
            self.assertEqual(image.size, (64, 48))
            self.assertEqual(first["historyFrames"], 0)

    @unittest.skipUnless(os.environ.get("MLXDLSS_NR_MODEL") and shutil.which("ffmpeg"), "Set MLXDLSS_NR_MODEL and install FFmpeg")
    def test_temporal_timeline_and_controls_reset_history_between_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=10:duration=0.8",
                            "-c:v", "libx264", str(source)], check=True)
            request = {"input": str(source), "video": True, "time": 0.4,
                       "options": ["--model", os.environ["MLXDLSS_NR_MODEL"], "--profile", "standard"]}
            bypass = {**request, "options": request["options"] + ["--intensity", "0"]}
            first_frame = {**request, "time": 0}
            run = subprocess.run([os.environ["MLXDLSS_NATIVE_PREVIEW_BINARY"], "preview-stream"],
                input="\n".join(map(json.dumps, [request, bypass, first_frame, request])) + "\n",
                text=True, capture_output=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stderr)
            first, bypass, start, again = map(json.loads, run.stdout.splitlines())
            self.assertEqual(first["historyFrames"], 3)
            self.assertAlmostEqual(first["time"], 0.4)
            self.assertEqual(start["historyFrames"], 0)
            self.assertEqual(bypass["processed"], bypass["original"])
            self.assertEqual(first["processed"], again["processed"])
