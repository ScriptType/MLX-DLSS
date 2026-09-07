# mlxdlss (Python)

PyTorch inference pipeline and bring-your-own-DLL weight tooling for the
recovered neural-rendering transformer of [MLX-DLSS](../README.md).

```sh
python -m pip install .            # numpy, torch, safetensors, pillow
python -m pip install '.[web,video]' # web UI and default temporal video (OpenCV)
python -m pip install '.[coreml]'  # optional Core ML conversion (macOS, Linux)

mlxdlss-weights all nvngx_dlssnr.dll weights/ --coreml 320x320   # your own DLL -> packed, logical, MLX, Core ML
mlxdlss-torch run --weights weights/dlssnr-weights-logical.safetensors --input in.png --output out.png
```

```python
from mlxdlss import NeuralRenderingPipeline
pipeline = NeuralRenderingPipeline.from_safetensors("weights/dlssnr-weights-logical.safetensors", device="auto")
result = pipeline.enhance(image_float32_hwc, profile="standard", processing_scale=2, detail_strength=2)
```

`precision="reference"` (default) computes in float32 with the recovered
E4M3/half rounding points; `"fast"` runs the same graph in float16 on CUDA or
MPS. No weights are included or downloaded: the DLL comes from your own NVIDIA
driver or Streamline package and every artifact stays on your machine.

New video jobs and `mlxdlss-video convert` enable temporal rendering by default.
`--no-temporal` keeps independent frames. `TemporalSession` uses bidirectional
DIS flow to reproject rendered history, rejects unreliable correspondence, and
resets on scene changes. Scales 1–4 work on both PyTorch and the rebuilt Metal
stream; history is retained before display detail enhancement. Existing saved
jobs retain their explicit settings. See the repository README for controls.
