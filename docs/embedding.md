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

## Performance expectations

Use the [paired measurements in the README](../README.md#accuracy-and-speed)
for the current Metal path. Processing scale increases the network's pixel
count quadratically; moving display composition and NR → FG onto Metal reduces
transfer and CPU work while preserving that model cost. Measure the complete
frame loop at the intended extent, precision and output format.
