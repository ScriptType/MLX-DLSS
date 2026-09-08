# Super resolution

**DLSS SR 2× for video and RTX VSR 2× for images are available in the native app,
web, CLI and Swift API.** Both are experimental and checked against their own
NVIDIA library with matching inputs and state.

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

In the web app on macOS 26+, add RTX VSR weights in **Settings**, select Auto
or Metal, then enable **Upscale 2×** on the Image page. It works alone or after
neural rendering, including live previews and batch export.

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

The native path restores **DLSS SDK 310.7.0, preset K, LDR, 2×**: feature
preparation, all eleven transformer blocks, reconstruction and recurrent
history. Swift/Metal runs every stage without Python at runtime.

```sh
.build/release/mlxdlss process-video in.mp4 --output out.mp4 --sr-model weights/dlss-sr.srmodel
```

In the app, enable **Upscale 2×** and choose **DLSS SR · Temporal**. In the web
app, set the DLSS SR model in Settings and enable **Upscale 2×** on Video.
Use MP4/MOV/M4V and Auto or Metal. SR runs after NR/FG; motion, scene resets,
timestamps and audio use the native video pipeline. Temporal is on by default;
when combined with NR, both share its history controls.

Swift callers set `MediaProcessingOptions.dlssSuperResolutionModel`. The lower
level `DLSSSuperResolver.upscale` accepts RGB and current-to-previous motion in
input-pixel units. A model instance owns one sequence; call `reset()` on seeks
and scene cuts. Ordinary video uses estimated optical flow, flat depth and
zero jitter. Game integration and other presets, scale factors and HDR remain
unsupported.

### Model preparation

This recovery still needs a **one-time CUDA capture from the original library**;
SR extraction directly from the library on a Mac is not implemented. The
capture contains first-frame launch parameters and buffers, including network
weights and the reconstruction filter. Convert these local artifacts with:

```sh
mlxdlss-weights package-sr libnvidia-ngx-dlss.so.310.7.0 CAPTURE_DIR weights/dlss-sr.srmodel
```

The converter verifies the library, weights and filter hashes, decodes logical
tensors and compiles the supplied reconstruction kernel to Metal. Generated
shader source, models and captures stay outside Git. Existing packages are
preserved. The prepared `.srmodel` can then be used entirely on the Mac.

### Reference checks

Against NVIDIA's Linux 310.7.0 implementation on RTX 4090, eight moving
512×384 → 1024×768 frames reached **59.8–72.2 dB PSNR**. Native preprocessing
matches the first two synthetic frames byte for byte; the third differs in
one FP16 value. The complete Swift API passes sequence and reset checks.

Small network rounding differences can select a different reconstruction
filter: rare channel errors reach **0.19** despite the low average error.
This establishes an experimental port, not bitwise or Windows DLL parity.
The reference suite is enabled with `MLXDLSS_SR_REFERENCE`; native media tests
also accept `MLXDLSS_SR_MODEL`. Broader video quality remains under evaluation.

The old Lanczos comparison did not establish port fidelity. Its harness also
used HDR flags for normalized sRGB and omitted the low-resolution-motion flag,
so it cannot justify the previous blanket rejection of DLSS SR.

See NVIDIA's
[DLSS integration guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS.md)
for motion, depth and jitter conventions.
