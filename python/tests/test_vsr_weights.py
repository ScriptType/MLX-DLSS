import hashlib
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import load_file

from mlxdlss.tools import extract_vsr_weights as vsr
from mlxdlss.tools.cli import main


class VSRWeightsTests(unittest.TestCase):
    def test_ohwi_conversion_and_zero_shortcut_bias(self):
        raw = np.arange(24, dtype=np.float16).reshape(2, 2, 2, 3)
        bias = np.array([0.25, -0.5], dtype=np.float16)
        blob = raw.tobytes() + bias.tobytes()
        tensors = vsr._decode(blob, {"conv": (2, 3, 2, 0, raw.nbytes), "skip": (2, 3, 2, 0, None)})
        np.testing.assert_array_equal(tensors["conv.weight"], raw.transpose(0, 3, 1, 2))
        np.testing.assert_array_equal(tensors["conv.bias"], bias)
        np.testing.assert_array_equal(tensors["skip.bias"], np.zeros(2, np.float16))
        with self.assertRaises(ValueError):
            vsr._decode(blob[:-1], {"conv": (2, 3, 2, 0, raw.nbytes)})
        with self.assertRaises(ValueError):
            vsr._decode(np.array([np.nan], np.float16).tobytes(), {"conv": (1, 1, 1, 0, None)})

    def test_cli_pins_source_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source = pathlib.Path(folder) / "library.so"
            output = pathlib.Path(folder) / "weights.safetensors"
            blob = np.array([0.5, 0.25], np.float16).tobytes()
            source.write_bytes(blob)
            with self.assertRaisesRegex(ValueError, "Unsupported VSR library"):
                vsr.extract(source)
            with patch.object(vsr, "SOURCE_SHA256", hashlib.sha256(blob).hexdigest()), \
                    patch.object(vsr, "LAYERS", {"conv": (1, 1, 1, 0, 2)}):
                self.assertEqual(main(["extract-vsr", str(source), str(output)]), 0)
                np.testing.assert_array_equal(load_file(output)["conv.bias"], [0.25])
                saved = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    main(["extract-vsr", str(source), str(output)])
                self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
