"""Extract RTX VSR High Bitrate Low 2x weights from a locally supplied library."""
from __future__ import annotations

import argparse
import hashlib
import pathlib

import numpy as np

SOURCE_SHA256 = "c7f2387a565e41b77a624634c102c2254d5285d28f886bd5db7491df8b1b037e"
MODEL = "rtx-vsr-1.8.2-highbitrate-low-2x"

# Cout, Cin, kernel size, OHWI weight offset, bias offset (None means zero).
# Offsets apply only to the SHA-256-pinned Linux VSR 1.8.2 library.
LAYERS = {
    "decoder0.conv0": (64, 128, 3, 32068352, 32068224),
    "decoder0.conv1": (64, 64, 3, 31994464, 31994336),
    "decoder0.shortcut": (64, 128, 1, 31977920, None),
    "decoder0.upsample": (64, 64, 3, 32215968, 32215840),
    "decoder1.conv0": (32, 64, 3, 31903936, 31903872),
    "decoder1.conv1": (32, 32, 3, 31885408, 31885344),
    "decoder1.shortcut": (32, 64, 1, 31881216, None),
    "decoder1.upsample": (32, 64, 3, 31940896, 31940832),
    "decoder2.conv0": (32, 64, 3, 31825728, 31825664),
    "decoder2.conv1": (32, 32, 3, 31807200, 31807136),
    "decoder2.shortcut": (32, 64, 1, 31803008, None),
    "decoder2.upsample": (32, 32, 3, 31862688, 31862624),
    "decoder3.conv0": (16, 32, 3, 31784416, 31784384),
    "decoder3.conv1": (16, 16, 3, 31779744, 31779712),
    "decoder3.shortcut": (16, 32, 1, 31778656, None),
    "decoder3.upsample": (16, 32, 3, 31793696, 31793664),
    "encoder0.conv0": (16, 16, 3, 32634464, 32634432),
    "encoder0.conv1": (16, 16, 3, 32629792, 32629760),
    "encoder0.shortcut": (16, 16, 1, 32629216, None),
    "encoder1.conv0": (32, 16, 3, 32619936, 32619872),
    "encoder1.conv1": (32, 32, 3, 32601408, 32601344),
    "encoder1.shortcut": (32, 16, 1, 32600288, None),
    "encoder2.conv0": (32, 32, 3, 32581760, 32581696),
    "encoder2.conv1": (32, 32, 3, 32563232, 32563168),
    "encoder2.shortcut": (32, 32, 1, 32561088, None),
    "encoder3.conv0": (64, 32, 3, 32524128, 32524000),
    "encoder3.conv1": (64, 64, 3, 32450240, 32450112),
    "encoder3.shortcut": (64, 32, 1, 32445984, None),
    "encoder4.conv0": (64, 64, 3, 32372096, 32371968),
    "encoder4.conv1": (64, 64, 3, 32298208, 32298080),
    "encoder4.shortcut": (64, 64, 1, 32289856, None),
    "output.conv": (64, 16, 3, 31760160, 31760032),
    "output.project": (48, 64, 1, 31753856, 31753760),
}


def _decode(blob: bytes, layers: dict) -> dict[str, np.ndarray]:
    tensors = {}
    for name, (cout, cin, size, weight_offset, bias_offset) in layers.items():
        weight = np.frombuffer(blob, "<f2", cout * cin * size * size, weight_offset)
        weight = weight.reshape(cout, size, size, cin).transpose(0, 3, 1, 2)
        bias = (np.zeros(cout, dtype=np.float16) if bias_offset is None
                else np.frombuffer(blob, "<f2", cout, bias_offset))
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError(f"Nonfinite VSR weights: {name}")
        tensors[name + ".weight"] = np.ascontiguousarray(weight)
        tensors[name + ".bias"] = np.ascontiguousarray(bias)
    return tensors


def extract(library: pathlib.Path) -> dict[str, np.ndarray]:
    blob = library.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Unsupported VSR library SHA-256: {digest}; expected {SOURCE_SHA256}")
    return _decode(blob, LAYERS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=pathlib.Path, help="libnvidia-ngx-vsr.so.1.8.2")
    parser.add_argument("destination", type=pathlib.Path, help="output .safetensors")
    args = parser.parse_args(argv)
    from safetensors.numpy import save

    tensors = extract(args.library)
    data = save(tensors, metadata={"model": MODEL, "source_sha256": SOURCE_SHA256, "layout": "OIHW"})
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("xb") as output:
        output.write(data)
    print(f"Wrote {len(tensors)} tensors to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
