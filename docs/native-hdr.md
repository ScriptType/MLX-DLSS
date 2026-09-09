# Native HDR frames

`NativeHDRVideoReader.nextDecoded()` returns retained decoder planes and metadata without importing; adapters can pass them directly to a shared GPU engine. `next()` composes this boundary with the session importer. The reader requests NV12 for 8-bit SDR and P010 for PQ/HLG or higher-bit-depth SDR from AVAssetReader. `MLXHDRImporter` binds both IOSurface planes through CVMetalTextureCache and converts them in a Metal compute pass. There is no BGRA8 intermediate. `MLXHDRFrame.original` is completed, immutable float32 NHWC RGB in linear BT.2020, in absolute cd/m² (nits). It retains both the decoder buffer and GPU allocation. Downstream work may start when `importFrame` returns; returned owners must remain retained until consumers complete. This is the original bypass and reconstruction boundary.

The importer expands full or limited code range (8-bit 16–235/16–240; 10-bit 64–940/64–960), reconstructs chroma with bilinear sampling at the declared progressive siting, applies the declared YCbCr matrix, decodes transfer, and converts linear primaries. P010 uses the high 10 bits of each 16-bit word. Hardware texture views handle plane strides. Source metadata includes exact CMTime PTS/duration, source/track/frame/generation identity, clean crop, preferred transform, pixel aspect, source colour tags, mastering display bytes and content-light bytes. Geometry remains in metadata for the presenter; it is not passed through SDR image conversion.

PQ uses the ST 2084 EOTF with 10,000-nit normalization. HLG uses inverse OETF followed by its luminance-dependent reference OOTF, with ideal black and gamma `1.2 + 0.42 log10(peak/1000)`; the default reference display peak is 1,000 nits and supported policy range is 400–2,000 nits. The fixed working-space result is display-linear. This HLG reference rendering is part of source interpretation; it must not be applied again by the display presenter. BT.709-tagged SDR uses the ideal-black BT.1886 display EOTF (power 2.4), while explicitly sRGB-tagged input uses the sRGB EOTF. SDR white maps to 203 nits by default. Linear source codes use the same reference-white scale. Missing source interpretation uses documented BT.709/BT.1886/progressive-left defaults recorded in `assumptions`; unsupported tags are errors.

`referenceWhiteNits` defaults to 203. It controls the SDR model proxy and eventual relative EDR normalization, never the maximum original or reconstructed value. The equations follow [ITU-R BT.2100](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-2-201807-S%21%21PDF-E.pdf) and the reference-white convention described in [BT.2408](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-3-2019-PDF-E.pdf). Chroma attachment interpretation follows [Apple's CoreVideo chroma locations](https://developer.apple.com/documentation/corevideo/image-buffer-chroma-location-constants).

## Neural processing and reconstruction

`NativeHDRProcessor` retains model/kernel resources and motion history. It converts the original BT.2020 image to a separate sRGB/BT.709 proxy, divided by reference white, with a luminance shoulder above 0.75 and bounded sRGB output. Optical flow and neural processing see only that proxy. Explicit processing width/height can differ from source size; neural output is resampled to original size before reconstruction. Changing those dimensions resets model history without reloading weights.

For BT.2020 working images, reconstruction computes linear BT.709 proxy/model luminances and a gain `clamp(modelY / proxyY, 0, maximumLuminanceRatio)`. It transforms the residual `modelRGB - proxyRGB * gain` into BT.2020 and adds it, scaled by colour strength, to `originalRGB * gain`. Effect strength mixes this result with the original. Proxy-black pixels keep their original; this avoids unstable division near black. The residual has zero BT.709 luminance before floating-point roundoff, so the gain limit controls luminance while colour strength controls chromatic change. Effect and colour strengths are 0–1, and the maximum gain is at least 1. The same equations run in the CPU reference and GPU codec.

Identity-model reconstruction preserves original wide-gamut values that the SDR proxy cannot represent. Zero strength returns the same original frame object, without proxy quantization or model execution. The original is never replaced by an inverse tone curve. No final HDR clamp is applied; signed out-of-gamut components and extended values survive float packing. The existing BT.709 codec mode retains its original algorithm for compatibility.

Each processed result exposes original, proxy, identity-model and enhanced views at the same exact timestamp, plus the retained source metadata. Original/identity/enhanced use linear BT.2020 nits. Proxy alone uses bounded sRGB. Display mapping and overlay composition belong after reconstruction. `MLXPixelBufferWriter(halfOutput: true)` writes unclipped RGBA16F; the 8-bit writer remains explicitly bounded SDR. `NativeMediaProcessor.processVideo` is still an SDR exporter via a deliberate proxy adapter and the existing BT.709-tagged writer, not an HDR cache format.

## Focused verification

`MLXHDRFrameTests` checks NV12/P010, limited/full range and grey samples against independent double-precision SDR/PQ/HLG equations, saturated PQ samples, source metadata, primary conversion and half output above reference white. `MLXNeuralRenderingDisplayCodecTests` compares CPU/GPU BT.2020 reconstruction, identity and highlights. Native HDR processor tests exercise exact original bypass, discontinuities, decoded clips and an optional real model sequence. These checks establish numeric development behavior; they do not establish HDR display accuracy or M5 throughput.

The development run on Apple M3/macOS 26.5 passed the focused HDR import/codec/processor checks plus existing CPU codec and NR/FG/VSR export regression. A three-frame real model sequence retained 960–965-nit highlights; identity reconstruction differed by at most 0.0000611 nits. Ten decoded PQ sample positions were independently reconstructed from FFmpeg-decoded YUV using double-precision equations and agreed with the native RGBA16F captures within half-float rounding (largest absolute difference 0.843 nits at approximately 2,171 nits). These are numeric checks, not target-hardware performance results.

To repeat the focused GPU/model checks from the parent workspace, select the configured Xcode environment, copy `vendor/MLX-DLSS/.build/release/mlx.metallib` to the release XCTest bundle's `Contents/MacOS`, and run:

```sh
MLXDLSS_NEURAL_RENDERING_PACKAGE="$PWD/models/neural-rendering/NeuralRendering.dlssmodel" \
MLXDLSS_HDR_FIXTURES="$PWD/assets/test-clips" \
MLXDLSS_FG_WEIGHTS="$PWD/models/framegen.safetensors" \
MLXDLSS_VSR_WEIGHTS="$PWD/models/vsr.safetensors" \
swift test --package-path vendor/MLX-DLSS -c release --jobs 2 \
  --filter 'MLXHDRFrameTests|MLXVideoFrameTests|MLXNeuralRenderingDisplayCodecTests|NativeHDRProcessorTests|NativeMediaProcessorTests|NeuralRenderingDisplayCodecTests'
```

Importer/writer GPU durations come from completed Metal command buffers and are optional when the driver does not expose timestamps. Processor proxy/motion/inference/reconstruction timings measure completed stage wall time, including waits; they are not labeled isolated GPU timings.
