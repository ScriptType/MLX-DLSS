# Super resolution

**RTX VSR 2× works in the native app, CLI and Swift API. DLSS Super Resolution
is still being recovered.** Each port is checked against its own NVIDIA library
with matching inputs and state.

NVIDIA's [browser setting](https://www.nvidia.com/content/Control-Panel-Help/vLatest/en-us/mergedProjects/Display/Reference_Adjust_Video_Image_Settings.htm)
upscales video playback. The VFX SDK also accepts individual image frames,
which makes its VSR model usable for stills.

## RTX Video Super Resolution

The experimental port supports VSR **1.8.2, High Bitrate Low (mode 16), 2×**.
It runs on Swift/MLX/Metal without Python at runtime. The mode processes each
image independently and uses RGB8 input/output quantization.

Prepare weights from your own `libnvidia-ngx-vsr.so.1.8.2`, supplied with
[NVIDIA VFX](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html):

```sh
mlxdlss-weights extract-vsr libnvidia-ngx-vsr.so.1.8.2 weights/vsr.safetensors
.build/release/mlxdlss process-image in.png --output out.png --vsr-weights weights/vsr.safetensors
.build/release/mlxdlss process-video in.mp4 --output out.mp4 --vsr-weights weights/vsr.safetensors
```

The converter checks the exact source SHA-256 and preserves existing output.
Other library builds, quality modes and scale factors are unsupported.

In the app, choose the weights under **Super Resolution** and enable **Upscale
2×**. Live preview uses the same model. VSR can run alone or after NR and FG;
timestamps, audio and frame-generation cadence are preserved. Swift callers
set `MediaProcessingOptions.superResolutionWeights`, or use
`VideoSuperResolver.upscale` for `[N,H,W,3]` tensors.

### Reference checks

Reference: NVIDIA's Linux VSR 1.8.2 library on RTX 4090, driver 580.173.02.
Metal: M2 Max, FP16. This does not establish Windows DLL parity.

| Inputs | Maximum channel difference |
| --- | --- |
| Synthetic frame, 640×360 still, eight 512×384 video frames | 1/255 |
| 97×65 noise, gradients and one-pixel lines | 1/255 |
| 97×65 black/white checkerboard | 2/255 in 6 of 75,660 values |
| 1920×1080 still → 3840×2160 | 2/255 in 5 of 24,883,200 values |

All 66 extracted tensors match the GPU-captured parameters byte for byte.
Model tests cover quantization, output activation, pixel shuffle, borders and
batching. Private reference captures are loaded through
`MLXDLSS_VSR_REFERENCE`; vendor data is excluded from Git.

The NVIDIA Python binding reports incorrect DLPack row strides for some odd
widths. The reference runner uses edge padding and crops the output; this was
checked against the original unpadded CUDA output surface.

## DLSS Super Resolution

The original Linux DLSS SDK 310.7.0 library now runs on RTX with corrected LDR
and low-resolution-motion flags. Captures include individual network stages,
static history and moving video. Reconstructing the Metal graph and temporal
state remains open; DLSS SR is not exposed in the app.

The old Lanczos comparison did not establish port fidelity. Its harness also
used HDR flags for normalized sRGB and omitted the low-resolution-motion flag,
so it cannot justify the previous blanket rejection of DLSS SR.

The next gate is matching the original preprocessing, transformer stages and
history update on identical inputs. Quality on finished media is a separate
check after that. See NVIDIA's
[DLSS integration guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS.md)
for motion, depth and jitter conventions.
