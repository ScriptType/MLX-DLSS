import pathlib
import hashlib
import json
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mlxdlss.tools import package_dlss_sr

from mlxdlss.tools.package_dlss_sr import (
    captured_buffer,
    decompress_lz4,
    package,
    postprocess,
)


class PackageDLSSSRTests(unittest.TestCase):
    def test_lz4_literals_and_overlapping_matches(self):
        self.assertEqual(decompress_lz4(b"\x50hello", 5), b"hello")
        self.assertEqual(decompress_lz4(b"\x32abc\x03\x00", 9), b"abcabcabc")
        self.assertEqual(decompress_lz4(b"\xf0\x05" + b"x" * 20, 20), b"x" * 20)

    def test_rejects_truncation_invalid_matches_and_size(self):
        for block, size in [
            (b"\xf0", 20),
            (b"\x50abc", 5),
            (b"\x10a\x00\x00", 5),
            (b"\x10a\x02\x00", 5),
            (b"\x10a\x01", 5),
            (b"\x50hello", 4),
            (b"\x10a", 3),
            (b"", 1 << 24),
        ]:
            with self.subTest(block=block, size=size), self.assertRaises(ValueError):
                decompress_lz4(block, size)

    def test_capture_uses_pointer_offset_and_rejects_missing_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            params = bytearray(176)
            struct.pack_into("<Q", params, 64, 0x1008)
            (root / "L0001_params.bin").write_bytes(params)
            (root / "L0001_buf1000.bin").write_bytes(b"padding!weights")
            self.assertEqual(captured_buffer(root, 1, 64, 7), b"weights")
            with self.assertRaises(ValueError):
                captured_buffer(root, 1, 64, 8)

    def test_unsupported_library_and_existing_destination(self):
        with self.assertRaises(ValueError):
            postprocess(b"unsupported library")
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            sentinel = root / "existing"
            sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                package(root / "missing-library", root / "missing-capture", root)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_package_preserves_shader_bytes_on_every_platform(self):
        shader = "// Local conversion fixture\n// END HEADER\nreturn;\n"
        packed = b"captured weights"
        coefficients = np.zeros(32768, dtype="<f2").tobytes()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            library = root / "library"
            library.write_bytes(b"fixture")
            destination = root / "model.srmodel"
            with (
                patch.object(package_dlss_sr, "postprocess", return_value=shader),
                patch.object(
                    package_dlss_sr,
                    "captured_buffer",
                    side_effect=[packed, coefficients],
                ),
                patch.object(
                    package_dlss_sr,
                    "decode_weights",
                    return_value={"fixture": np.zeros(1, dtype=np.float16)},
                ),
                patch.object(
                    package_dlss_sr,
                    "WEIGHTS_SHA256",
                    hashlib.sha256(packed).hexdigest(),
                ),
                patch.object(
                    package_dlss_sr,
                    "FILTER_SHA256",
                    hashlib.sha256(coefficients).hexdigest(),
                ),
                patch.object(
                    package_dlss_sr,
                    "METAL_SHA256",
                    hashlib.sha256(shader.encode()).hexdigest(),
                ),
            ):
                package(library, root, destination)
            actual = (destination / "postprocess.metal").read_bytes()
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(actual, shader.encode("utf-8"))
            self.assertEqual(
                hashlib.sha256(actual).hexdigest(), manifest["postprocessSHA256"]
            )


if __name__ == "__main__":
    unittest.main()
