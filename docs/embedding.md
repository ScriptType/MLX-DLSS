# Embedding the neural-rendering component

The `mlxdlss` commands are thin wrappers around these types; an application, a
media pipeline or a server drives the recovered checkpoint through the
library alone. Every example assumes a user-supplied `MODEL.dlssmodel` produced
by `mlxdlss-weights`; MLX-DLSS never bundles or redistributes weights.

| Use case | Type | Cadence |
| --- | --- | --- |
| Still image, screenshot, video frames without history | `NeuralRenderingFirstFrameBackend` | frame independent |
| Frames with motion vectors, depth and display history | `NeuralRenderingTemporalReferenceBackend` | consecutive frames |
| Raw float32 tensors from any host | either backend around a `NeuralRenderBackend` head | as above |

Both backends conform to `NeuralRenderBackend`, so a host swaps the head
(`MLXNeuralRenderer` or `CoreMLNeuralRenderer`) or the temporal path without
touching its own frame loop.

## Network geometry

The checkpoint runs on a network extent of at least `320×320` and multiples of
`64`. `NeuralRenderingNetworkGeometryPolicy` maps a logical frame onto it:
`.vendorAligned` (default) mirrors the frame to the right and bottom, runs the
head on the aligned extent and crops the result, so `1920×1080` runs at
`1920×1088`; `.matchOutput` feeds the frame unchanged and is only for sizes the
head accepts directly.

## Still frame

```swift
import DLSSCore
import DLSSMLX

let head = try MLXNeuralRenderer(
  packageURL: modelPackageURL, executionMode: .metalFused, computePrecision: .float16
)
var configuration = NeuralRenderingFirstFrameConfiguration(profile: .standard)
configuration.intensity = 1
configuration.featureControls = NeuralRenderingFeatureControls(
  normalizedStyle: 0, localToneStrength: 1, localStructureStrength: 1
)
let backend = NeuralRenderingFirstFrameBackend(head: head, configuration: configuration)

let color = try HostTensor(
  descriptor: TensorDescriptor(name: "color", shape: [1, height, width, 3], dataType: .float32, layout: .nhwc),
  bytes: rgbFloat32Data
)
let result = try await backend.render(NeuralRenderRequest(sequenceID: 1, inputs: [color]))
let rgb = result.output(named: "color")!.bytes   // [1, height, width, 3] float32 in [0, 1]
```

An optional `controlMask` tensor (`[1, H, W, 3]`, red = blend, green = tone,
blue = structure) travels next to `color` in the same request. The photoreal
recipe (`--processing-scale`, `--detail-strength`, `--colour-strength`) is
`NeuralRenderingDetailComposition.resample` and `.compose` applied around the
backend exactly as `mlxdlss run` and `mlxdlss render-image` do.

## Temporal reference

```swift
let temporal = NeuralRenderingTemporalReferenceBackend(
  backend: head, depthInverted: false, controlMaskIntensity: 1,
  temporalPreprocessor: try MetalNeuralRenderingTemporalFeaturePreprocessor(depthGuideMode: .flat)
)
let request = try NeuralRenderRequest(
  sequenceID: 1,
  inputs: [color, motion, depth],   // motion: [1, H, W, 2] normalized history-UV offsets, depth: [1, H, W, 1]
  temporalContext: NeuralRenderFrameContext(streamID: 1, frameIndex: frameIndex)
)
let output = try await temporal.render(request).output(named: "color")
```

Consecutive `frameIndex` values keep the display history; a gap, a stream
change or a reset request clears it. `NeuralRenderingTemporalReferencePreprocessor.normalizePixelMotion`
converts engine pixel motion with its scale and jitter.

## Metal video output and frame generation

`MLXNeuralRenderingDeviceTemporalBackend` can compose display RGB and generate
intermediate frames inside its actor, keeping the float32 NR → FG arrays on
the GPU. The portable `render()` method still returns the original full-size
`HostTensor`; opt into the streaming output separately:

```swift
import Foundation

let video = try MLXNeuralRenderingDeviceTemporalBackend(
  packageURL: modelPackageURL, executionMode: .metalFused, computePrecision: .float16,
  videoOutput: MLXVideoOutputOptions(width: outputWidth, height: outputHeight,
                                   detailStrength: 2, format: .u8),
  frameGeneration: MLXVideoFrameGenerationOptions(weightsURL: frameGenerationWeightsURL,
                                                precision: .float16, factor: 2, batch: 4)
)
// Call sequentially for each input frame; request contains processing-size RGB and guides.
let written = try await video.renderToStream(request, source: originalColor, to: destination)
// Once, at end of input, emit any remaining generated/original pairs.
let finalWritten = try await video.finishStream(to: destination)
```

`originalColor` is a float32 NHWC `HostTensor` at the configured output extent;
`destination` is a writable `FileHandle`. The output is raw RGB in the selected
`.f32`, `.u8`, or `.u16` format. Omit `frameGeneration` for one display frame per
input. With FG, calls can write zero or several frames: one first original,
then batches of generated/original frames in order. Keep one actor per video,
and drain `finishStream` before closing the output. Temporal resets affect NR
history while retaining the consecutive FG pairs, matching the video adapter.

## Fixed-shape Core ML head

```swift
import DLSSCoreML

let head = try await CoreMLNeuralRenderer(
  modelURL: coreMLPackageURL,    // converted at the network extent, e.g. 320×320
  configuration: CoreMLBackendConfiguration(computeUnits: .cpuAndGPU)
)
```

The package is compiled for one network extent: a `256×256` frame needs a
`320×320` package, `1080p` needs `1920×1088`. A Core ML package freezes the
graph at conversion time, so rebuild it with `mlxdlss-weights coreml` whenever the
recovered graph changes; the MLX path applies such fixes at load time.

## Native media and live preview

On macOS 26+, `DLSSMedia` owns AVFoundation decode/encode, VideoToolbox or Vision
motion, temporal NR and FG. It requires no Python process or raw-frame pipe:

```swift
import DLSSMedia

var options = MediaProcessingOptions(renderingModel: modelPackageURL,
                                     frameGenerationWeights: frameGenerationWeightsURL)
options.detailStrength = 2
let processor = NativeMediaProcessor()
let result = try await processor.processVideo(input: inputURL, output: outputURL,
                                              options: options) { progress in
  // Update UI on its actor; keep this callback short.
  print(progress.inputFrames, progress.outputFrames)
}
```

For images, omit FG weights and call `processImage`. For a custom frame loop,
`MLXVideoFrame` imports IOSurface-backed pixel buffers and
`MLXPixelBufferWriter` exports them; `renderVideoFrame` retains temporal history
and applies display settings without a host float32 copy. Submit one frame at
a time per renderer. Cancellation prevents publication of a partial output;
existing files are never replaced.

`NativeMediaPreview.render(MediaPreviewRequest(input:isVideo:time:options:))`
returns original/processed `CGImage`s, the actual selected timestamp and timing.
Retain one preview actor to reuse weights and decoded frames. Submit requests
sequentially, coalesce changes and discard superseded responses, as the native
app does. Video preview uses up to three preceding frames and a fresh temporal
history per request; FG is reserved for export. Preview and export should share
one scheduling lane so they do not compete for the GPU.

`mlxdlss preview-stream` exposes that actor to the web UI on macOS 26+. Each
JSON line contains `input`, `video`, `time` and an `options` array of CLI flags.
Each response contains base64 PNGs (`original`, `processed`), dimensions,
`time`, `duration`, `frameInterval`, `historyFrames` and `elapsedSeconds`.
Errors return an `error` field; the process remains available for the next request.

The native `process-video` command exposes these options in addition to the
[rendering controls](../README.md#cli):

| Options | Behavior |
| --- | --- |
| `--temporal on\|off`, `--motion automatic\|vision\|videotoolbox\|zero` | History and motion backend; Automatic falls back to Vision if VT cannot start |
| `--scene-cut-threshold 0.3` | Reset unreliable history; 0 disables automatic resets |
| `--start-frame N`, `--frames N` | Select an input range |
| `--codec h264\|hevc\|prores`, `--bitrate BPS` | Encoder; ProRes requires MOV |
| `--factor 2`, `--order nr-fg\|fg-nr`, `--slow-motion on` | FG cadence/order; slow motion preserves audio pitch |
| `--audio off` | Omit audio |

Original timestamps and variable frame intervals are retained. FG emits
`(N-1)×factor+1` frames. JSON reports stage timings; stderr reports progress.
Native video is SDR 8-bit sRGB; PNG/TIFF exports retain 16 bits. The Python
adapter supports custom FFmpeg arguments and RGB16 video.

## Performance

M2 Max, 38-core GPU, real weights. Processing scale increases network pixel
count quadratically. Compare complete frame loops at the same scale and output
format; warm kernel timings exclude media I/O and startup.

Same 228-frame 512×384, 60 fps clip, temporal rendering, detail 2:

| Path | Whole clip | Input frames/s |
| --- | --- | --- |
| Before the earlier optimizations | 18.4–22.8 s | 10.0–12.4 |
| Earlier optimized Python/Metal pipeline | 10.1–16.1 s | 14.2–22.6 |
| Native pipeline | 11.33–12.36 s | 18.4–20.1 |

Desktop load varied across days; these ranges are not a paired speedup claim.
Earlier alternating runs showed about 1.4× throughput. The latest four native
A/B runs showed another 4–9% with global attention/graph caching enabled, versus
11.77–13.43 s disabled. All four decoded RGB streams had the same SHA-256.
They ran without an executable search path. Decode/encode took 0.37–0.48 s,
motion 3.19–3.32 s, NR 7.06–7.73 s; peak process footprint was 1.54–1.81 GB.

Short global attention sequences use on-chip tiles and cached MLX graphs while
preserving full spatial context, half denominators and E4M3 publication.
Before graph caching, eight blocks took 3.79 vs 4.85–5.51 ms at 48 tokens, and
21.59 vs 24.26–24.92 ms at 510 tokens, with exact output matches. Larger
sequences retain materialized attention. `MLXDLSS_STREAMED_GLOBAL_ATTENTION=0`
and `MLXDLSS_COMPILE_GLOBAL=0` disable the optimizations; forcing streamed
attention with `=1` above 512 tokens saves memory but was slower on this GPU.
See [NR/FG chain](frame-generation.md#gpu-video-chain) and
[FG output-head fusion](frame-generation.md#native-output-head-fusion) for
warm timings, including the additional 18–22% FG throughput gain.

Memory: the earlier Metal still-image path used 2.2 GB resident at 4K.
PyTorch float32 used about 1 GB per megapixel of network input; `fast` precision
roughly halves it. Bounded chunks prevent window count from growing the peak
(`MLXDLSS_TORCH_CHUNK_TOKENS`, 0 disables). PyTorch was within 0.002 MAE of Metal;
Core ML was within 0.008–0.014 MAE of the vendor captures.

Experiments excluded from the default path:

- **FP8 packing:** exact weight decode/GEMM took 0.54–2.90 ms vs MLX's
  0.25–0.84 ms. Packing window intermediates halved storage but gave no speed
  gain: three 1088×1920 blocks took 50.97 vs 50.46–50.61 ms.
- **ANE:** direct [ANEForge](https://github.com/sbryngelson/ANEForge) dispatch
  of the 1024→4096 projection took 0.30/0.51/1.04 ms at 48/192/768 tokens vs
  Metal's 0.27/0.35/0.73 ms, excluding copies; max half-output error was 0.001.
  Core ML chose ANE at 192 tokens (1.05 ms including Python prediction) and CPU
  at 48. These layer probes do not validate the whole network on ANE.
- **Temporal feature reuse:** updating the global latent every other frame
  added 0.0037–0.0072 mean RGB error, max 0.064, on 16 frames. A linear correction
  did not help. This used fresh encoder inputs and independent-frame rendering;
  a learned replacement needs a separate training corpus and temporal quality gate.
