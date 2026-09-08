"""Prepare a native DLSS SR package from a local library and CUDA parameter capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np
from safetensors.numpy import save_file

from .ptx_metal import compile_kernel

SOURCE_SHA256 = "fc19b68cefb4218e0954fab812c782e0aa4d30526e0151c796e8887e0eca3acb"
POST_SHA256 = "5133a026f4355aacd611ee4886a71848cc3e7c5cad4f283ef40ea27ecce4f1de"
METAL_SHA256 = "1bd949170193f077f75f9c08f8a4a9a6abf21088efbd7795ad8cd463be3517ae"
WEIGHTS_SHA256 = "ddaa59cd82deed53bad9d65b82b454f1439fc4875e4220b3ffafafb6cd0296a1"
FILTER_SHA256 = "88aa97aa1e2cb9e2e7c932476b24793a4fb998a97c76c3b351212224bc00ab3c"
MODEL = "dlss-sr-310.7.0-k-ldr-2x"
CHANNELS = [32, 64, 64, 96, 128, 160, 128, 96, 64, 64, 32]
HEADS = [2, 2, 2, 4, 4, 8, 4, 4, 2, 2, 2]
OFFSETS = [
    0,
    67264,
    215872,
    380928,
    759552,
    1384576,
    2189952,
    2782848,
    3161984,
    3327360,
    3476352,
    3545824,
]


def decompress_lz4(data: bytes, size: int) -> bytes:
    """Decode a bounded raw LZ4 block, as stored in CUDA fatbinary entries."""
    if size < 0 or size > 1 << 20:
        raise ValueError("Invalid PTX block size")
    output = bytearray()
    cursor = 0

    def length(value: int) -> int:
        nonlocal cursor
        if value == 15:
            while True:
                if cursor >= len(data):
                    raise ValueError("Truncated LZ4 length")
                extra = data[cursor]
                cursor += 1
                value += extra
                if extra != 255:
                    break
        return value

    while cursor < len(data):
        token = data[cursor]
        cursor += 1
        count = length(token >> 4)
        if cursor + count > len(data) or len(output) + count > size:
            raise ValueError("Invalid LZ4 literal")
        output.extend(data[cursor : cursor + count])
        cursor += count
        if cursor == len(data):
            break
        if cursor + 2 > len(data):
            raise ValueError("Truncated LZ4 match")
        distance = int.from_bytes(data[cursor : cursor + 2], "little")
        cursor += 2
        count = length(token & 15) + 4
        if distance == 0 or distance > len(output) or len(output) + count > size:
            raise ValueError("Invalid LZ4 match")
        for _ in range(count):
            output.append(output[-distance])
    if len(output) != size:
        raise ValueError("LZ4 output size mismatch")
    return bytes(output)


def postprocess(library: bytes) -> str:
    if hashlib.sha256(library).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            "Unsupported DLSS SR library; requires the supported Linux 310.7.0 build"
        )
    # The exact library digest pins the entry header, codec and supported variant.
    entry = 25420872
    header_size = struct.unpack_from("<I", library, entry + 4)[0]
    payload = library[entry + header_size : entry + header_size + 44810]
    ptx = decompress_lz4(payload, 149075).decode("utf-8")
    if hashlib.sha256(ptx.encode()).hexdigest() != POST_SHA256:
        raise ValueError("Unexpected DLSS SR postprocess source")
    metal = compile_kernel(ptx)
    if hashlib.sha256(metal.encode()).hexdigest() != METAL_SHA256:
        raise ValueError(
            "DLSS SR shader conversion changed; verify reference parity before packaging"
        )
    return metal


def captured_buffer(capture: Path, launch: int, parameter: int, size: int) -> bytes:
    params = (capture / f"L{launch:04}_params.bin").read_bytes()
    if len(params) != (176 if launch == 1 else 328):
        raise ValueError("Unexpected CUDA parameter layout")
    address = struct.unpack_from("<Q", params, parameter)[0]
    for path in capture.glob(f"L{launch:04}_buf*.bin"):
        match = re.fullmatch(rf"L{launch:04}_buf([0-9a-f]+)\.bin", path.name)
        if not match:
            continue
        base = int(match[1], 16)
        if base <= address and address + size <= base + path.stat().st_size:
            with path.open("rb") as source:
                source.seek(address - base)
                data = source.read(size)
            if len(data) != size:
                raise ValueError("Truncated CUDA buffer")
            return data
    raise ValueError(f"Missing CUDA buffer for launch {launch}, parameter {parameter}")


def decode_weights(blob: bytes) -> dict[str, np.ndarray]:
    if len(blob) != OFFSETS[-1]:
        raise ValueError("Unexpected SR packed weight size")
    tensors = {}

    def matrix(offset, k, n, kind="b", chunk=None):
        result = np.empty((k, n), dtype=np.float16)
        for j in range(n // 16):
            for i in range(k // 16):
                index = (
                    j * (k // 16) + i
                    if chunk is None
                    else (i // chunk) * (n // 16) * chunk + j * chunk + i % chunk
                )
                tile = np.frombuffer(
                    blob, dtype="<f2", count=256, offset=offset + index * 512
                ).reshape(32, 4, 2)
                for r in range(4):
                    for lane in range(32):
                        g, t = divmod(lane, 4)
                        for half in range(2):
                            row, col = (
                                (g + (r % 2) * 8, t * 2 + (r // 2) * 8 + half)
                                if kind == "a"
                                else (t * 2 + (r % 2) * 8 + half, g + (r // 2) * 8)
                            )
                            result[i * 16 + row, j * 16 + col] = tile[lane, r, half]
        return result

    for index, (c, heads) in enumerate(zip(CHANNELS, HEADS)):
        stage = index + 1
        cursor = OFFSETS[index]

        def take(name, shape, kind="b", chunk=None):
            nonlocal cursor
            tensors[f"{stage}.{name}"] = (
                np.frombuffer(blob, "<f2", shape[0], cursor).copy()
                if len(shape) == 1
                else matrix(cursor, *shape, kind, chunk)
            )
            cursor += int(np.prod(shape)) * 2

        if stage == 1:
            take("embeddingBias", (32,))
            take("embedding", (16, 32))
        elif stage >= 7:
            take("embedding", (CHANNELS[index - 1], 4 * c))
            take("embeddingBias", (4 * c,))
        take("norm", (c,))
        for head in range(heads):
            for q, name in enumerate(["q", "k", "v"]):
                tensors[f"{stage}.{name}{head}"] = np.concatenate(
                    [
                        matrix(cursor + part * 6144 + q * 2048, 32, 32)
                        for part in range(c // 32)
                    ]
                )
            cursor += c * 32 * 3 * 2
        for head in range(heads):
            take(f"position{head}", (64, 64), "a")
        take("attentionBias", (c,))
        for head in range(heads):
            take(f"projection{head}", (32, c))
        take("ffScale", (c,))
        take("ffOutputBias", (c,))
        take("ffUp", (c, 4 * c))
        take("ffBias", (4 * c,))
        take("ffDown", (4 * c, c), chunk=2)
        if stage <= 5:
            padded = (
                (CHANNELS[index + 1] + 16 * heads - 1) // (16 * heads) * (16 * heads)
            )
            take("merge", (4 * c, padded))
            take("mergeBias", (padded,))
        elif stage == 11:
            take("outputBias", (48,))
            take("output", (32, 48))
        if cursor != OFFSETS[index + 1]:
            raise ValueError(f"SR stage {stage} weight layout mismatch")
    if not all(np.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Nonfinite SR weights")
    return tensors


def package(library: Path, capture: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    metal = postprocess(library.read_bytes())
    packed = captured_buffer(capture, 1, 64, OFFSETS[-1])
    coefficients = captured_buffer(capture, 12, 280, 65536)
    if (
        hashlib.sha256(packed).hexdigest() != WEIGHTS_SHA256
        or hashlib.sha256(coefficients).hexdigest() != FILTER_SHA256
    ):
        raise ValueError(
            "CUDA capture does not contain the supported SR weights and reconstruction filter"
        )
    weights = decode_weights(packed)
    table = np.frombuffer(coefficients, "<f2").copy()
    if not np.isfinite(table).all():
        raise ValueError("Nonfinite reconstruction filter")
    weights["reconstructionFilter"] = table
    # Create exclusively after validation; never replace an existing model package.
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    save_file(weights, destination / "weights.safetensors")
    (destination / "postprocess.metal").write_bytes(metal.encode("utf-8"))
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "format": "mlxdlss-sr-v1",
                "model": MODEL,
                "postprocessSHA256": METAL_SHA256,
                "sourceSHA256": SOURCE_SHA256,
            },
            indent=2,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "library", type=Path, help="locally supplied libnvidia-ngx-dlss.so.310.7.0"
    )
    parser.add_argument(
        "capture",
        type=Path,
        help="CUDA launch parameters and buffers for SR preset K, first frame",
    )
    parser.add_argument("destination", type=Path, help="output .srmodel directory")
    args = parser.parse_args(argv)
    package(args.library, args.capture, args.destination)
    print(f"Prepared native DLSS SR package: {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
