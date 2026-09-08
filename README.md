# MLX-DLSS

> [!TIP]
> **A note for the NVIDIA reader.** This port was worked out on a laptop and on
> GPU instances rented by the hour, some of which even booted. A pair of DGX
> Sparks would have replaced the rentals and would have a steady job here:
> experiments like this one, and the pet projects queued behind it. Hit me up on
> X: [@WaveCut](https://x.com/WaveCut).

Run NVIDIA's DLSS neural rendering and frame generation on images and video.
Apple Silicon uses MLX and Metal; PyTorch supports other GPUs and CPUs.

This is an experimental port, not a game integration or an NVIDIA product.
Bring your own NVIDIA libraries to extract weights. No vendor binaries or
weights are included or downloaded.

![Neural rendering: input, defaults, scale 2 with detail 2](docs/assets/neural-rendering-control.png)

[Frame-generation comparison](https://github.com/user-attachments/assets/ce94f426-910b-4556-bdf9-662cbdd5933a)

## Choose a mode

| Mode | Runtime | Preview |
| --- | --- | --- |
| macOS app | Swift, Metal, AVFoundation | Live settings; select a video frame on the timeline |
| Web UI | Python, NiceGUI, FFmpeg, OpenCV; Metal or PyTorch inference | Compare completed jobs |
| CLI / Python API | Native Metal or PyTorch | File output |

The web UI shares Metal kernels through the updated Swift binary. Live settings
previews are app-only; the native media pipeline serves the app and CLI.
`mlxdlss-web --native` is a separate pywebview wrapper and still needs Python.

## macOS quick start

Requires Apple Silicon, macOS 26+, Xcode with Swift 6.2+, CMake and Ninja.

```sh
scripts/build-native-app.sh
open '.build/MLX DLSS.app'
```

Choose your [prepared weights](#weights), import media, adjust settings and
start the queue. The app includes its CLI and Metal library. It needs no Python
or FFmpeg at runtime.

Temporal rendering is on by default. Live preview uses up to three preceding
frames; export uses the full sequence and applies frame generation. Native
video output is SDR 8-bit; PNG/TIFF stills retain 16-bit output.

## Weights

Initial extraction uses Python 3.10+:

```sh
python3 -m pip install ./python
mlxdlss-weights all nvngx_dlssnr.dll weights/
mlxdlss-weights extract-fg libnvidia-ngx-dlssg.so.310.7.0 weights/framegen.safetensors
```

Supported sources: `nvngx_dlssnr.dll` version **310.8.0.0** and frame generation
from **DLSS SDK 310.7.0**. `mlxdlss-weights sha256 FILE` checks the DLL build.

Outputs: `NeuralRendering.dlssmodel` for Metal,
`dlssnr-weights-logical.safetensors` for PyTorch and `framegen.safetensors` for
either backend.

## CLI

The app build also creates `.build/release/mlxdlss`:

```sh
.build/release/mlxdlss process-image in.png --output out.png --model weights/NeuralRendering.dlssmodel
.build/release/mlxdlss process-video in.mp4 --output out.mp4 --model weights/NeuralRendering.dlssmodel
.build/release/mlxdlss process-video in.mp4 --output out.mp4 --framegen-weights weights/framegen.safetensors --factor 2
```

Pass both weight options to combine effects; `--order nr-fg|fg-nr` selects their
order. Video defaults to temporal rendering, automatic optical flow, H.264 and
audio. Existing output files are preserved. [More options and APIs](docs/embedding.md#native-media-and-live-preview).

| Rendering control | Default | Effect |
| --- | --- | --- |
| `--profile` | `standard` | `standard`, `natural`, `cinematic`, `neutral` |
| `--processing-scale` | `1` | More detail at higher cost; range 1–4 |
| `--detail-strength` | `1` | Strength of fine changes; range 0–8 |
| `--colour-strength` | `1` | Strength of broad colour changes; range 0–4 |
| `--intensity` | `1` | Overall effect strength; 0 bypasses it |

## Web and Python

Requires Python 3.10+ and FFmpeg/ffprobe for video:

```sh
python3 -m pip install './python[web,video]'
mlxdlss-web                       # http://127.0.0.1:8181
```

Set weight paths and backend in **Settings**. The web UI supports images,
video effect chains, queues, cancellation and downloads. Temporal rendering
defaults to on for new videos; saved settings are kept.

For the web's Metal backend (image/tensor APIs support macOS 14+):

```sh
swift build -c release
scripts/prepare-mlx-metallib.sh "$(swift build -c release --show-bin-path)"
```

Repeat after a clean build to place `mlx.metallib` beside the binary.
For custom FFmpeg filters, RGB16 video, PyTorch, Core ML conversion and the
Python API, see the [Python guide](python/README.md).

## Accuracy and speed

| Measurement | Result |
| --- | --- |
| Neural rendering vs NVIDIA | 0.004–0.005 MAE on 1152–1408 px game renders |
| Frame generation vs NVIDIA | 59.9 dB PSNR; maximum difference 3/255 |
| Native temporal video, M2 Max | 18.4–20.1 input frames/s |

The video sample has 228 frames at 512×384, detail strength 2: **11.33–12.36 s**
including startup, motion, decode and encode. Desktop load affects timings.
See [measurements and experiments](docs/embedding.md#performance),
[FG benchmarks](docs/frame-generation.md#speed) and [parity limits](docs/recovery-notes.md).

## Docs and development

- [Swift API and native media](docs/embedding.md)
- [Frame generation](docs/frame-generation.md)
- [Super resolution: measured, not ported](docs/super-resolution.md)
- [Model recovery](docs/recovery-notes.md) and [research](docs/research/)

```sh
scripts/verify.sh                 # tests, builds and public-tree audit
```

CI tests Swift/Metal on Apple Silicon and Python on macOS, Linux and Windows.
Keep weights and captures outside Git. See [Contributing](CONTRIBUTING.md),
[Security](SECURITY.md), [Publication](PUBLICATION.md) and [Notice](NOTICE).

Source: [Apache 2.0](LICENSE). Extracted models retain the vendor's terms and
must not be redistributed.
