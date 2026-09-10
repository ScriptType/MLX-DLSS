// SEPARABLE-MOTION-IMPORT-BEGIN
@testable import DLSSMLX
// SEPARABLE-MOTION-IMPORT-END
import CoreVideo
import Foundation
import MLX

/// Immutable motion and confidence at the source extent, with current-to-previous
/// UV offsets. Only scalar scene-cut statistics are read back from the GPU.
public final class SeparableErosionVideoMotion: @unchecked Sendable {
  let vectors: MLXArray
  let confidence: MLXArray
  public let reset: Bool
  public let reliableFraction: Float
  public let warpedLumaError: Float

  public enum Units: Sendable { case sourcePixels, flowPixels, normalizedUV }

  public init(current: MLXVideoFrame, previous: MLXVideoFrame,
              backward: MLXPixelBuffer, forward: MLXPixelBuffer,
              units: Units, sceneCutThreshold: Float = 0.3) throws {
    guard current.width == previous.width, current.height == previous.height,
      sceneCutThreshold.isFinite, sceneCutThreshold >= 0 else {
      throw MLXMediaError("Motion frames must have equal extents and a finite nonnegative cut threshold")
    }
    func importFlow(_ buffer: MLXPixelBuffer) throws -> MLXArray {
      let format = CVPixelBufferGetPixelFormatType(buffer.buffer)
      guard format == kCVPixelFormatType_TwoComponent16Half || format == kCVPixelFormatType_TwoComponent32Float else {
        throw MLXMediaError("Optical flow must contain two float16 or float32 channels")
      }
      let dtype: DType = format == kCVPixelFormatType_TwoComponent16Half ? .float16 : .float32
      let storage = try MLXVideoFrame.storage(buffer.buffer, dtype: dtype)
      let params = MLXArray([UInt32(buffer.width), UInt32(buffer.height),
        UInt32(CVPixelBufferGetBytesPerRow(buffer.buffer) / dtype.size), UInt32(current.width),
        UInt32(current.height), units == .sourcePixels ? 1 : units == .flowPixels ? 2 : 0, 0, 0])
      let count = current.width * current.height * 2
      return Self.importFlow([storage, params], grid: (count, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [[1, current.height, current.width, 2]], outputDTypes: [.float32])[0]
    }
    let rawVectors = try importFlow(backward)
    // Invalid estimates still reject history below, but cannot enter a texture
    // address calculation even when scene-cut detection is disabled.
    vectors = which(isFinite(rawVectors), rawVectors, MLXArray(Float(0)))
    let reverse = try importFlow(forward)
    let count = current.width * current.height
    let params = MLXArray([UInt32(current.width), UInt32(current.height), 0, 0, 0, 0, 0, 0])
    let quality = Self.assess([current.array, previous.array, rawVectors, reverse, params],
      grid: (count, 1, 1), threadGroup: (256, 1, 1),
      outputShapes: [[1, current.height, current.width, 1], [1, current.height, current.width, 1]],
      outputDTypes: [.float32, .float32])
    // SEPARABLE-MOTION-EROSION-BEGIN
    confidence = SeparableMotionErosion.apply(quality[0], params: params)
    // SEPARABLE-MOTION-EROSION-END
    let statistics = stacked([(confidence .> 0.5).asType(.float32).mean(), quality[1].mean()])
    eval(vectors, confidence, statistics)
    let values = statistics.asArray(Float.self)
    reliableFraction = values[0]
    warpedLumaError = values[1]
    reset = sceneCutThreshold > 0 &&
      ((reliableFraction < 0.5 && warpedLumaError > sceneCutThreshold) ||
       (reliableFraction < 0.15 && warpedLumaError > 0.12))
  }

  static func resize(_ input: MLXArray, width: Int, height: Int, nearest: Bool) -> MLXArray {
    if input.dim(2) == width && input.dim(1) == height { return input }
    let params = MLXArray([UInt32(input.dim(2)), UInt32(input.dim(1)), UInt32(input.dim(3)),
      UInt32(width), UInt32(height), 0, 0, 0])
    let count = width * height * input.dim(3)
    return resizeKernel([input, params], template: [("nearest", nearest)],
      grid: (count, 1, 1), threadGroup: (256, 1, 1),
      outputShapes: [[1, height, width, input.dim(3)]], outputDTypes: [.float32])[0]
  }

  private static let sampleHeader = #"""
    template <typename Pointer>
    float read_image(Pointer image, float x, float y, uint c, uint channels, uint width, uint height) {
      x = clamp(x, 0.0f, float(width - 1)); y = clamp(y, 0.0f, float(height - 1));
      uint x0 = uint(x), y0 = uint(y), x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
      float a = x - float(x0), b = y - float(y0);
      float top = mix(image[(y0 * width + x0) * channels + c], image[(y0 * width + x1) * channels + c], a);
      float bottom = mix(image[(y1 * width + x0) * channels + c], image[(y1 * width + x1) * channels + c], a);
      return mix(top, bottom, b);
    }
    """#

  private static let importFlow = MLXFast.metalKernel(
    name: "mlxdlss_separable_motion_flow_import", inputNames: ["flow", "params"], outputNames: ["vectors"],
    source: #"""
      uint i = thread_position_in_grid.x, width = params[3], height = params[4];
      if (i >= width * height * 2) return;
      uint c = i % 2, x = (i / 2) % width, y = i / (2 * width);
      float fx = clamp((float(x) + 0.5f) * float(params[0]) / float(width) - 0.5f, 0.0f, float(params[0] - 1));
      float fy = clamp((float(y) + 0.5f) * float(params[1]) / float(height) - 0.5f, 0.0f, float(params[1] - 1));
      uint x0 = uint(fx), y0 = uint(fy), x1 = min(x0 + 1, params[0] - 1), y1 = min(y0 + 1, params[1] - 1);
      float top = mix(float(flow[y0 * params[2] + x0 * 2 + c]), float(flow[y0 * params[2] + x1 * 2 + c]), fx - float(x0));
      float bottom = mix(float(flow[y1 * params[2] + x0 * 2 + c]), float(flow[y1 * params[2] + x1 * 2 + c]), fx - float(x0));
      float divisor = params[5] == 1 ? float(c == 0 ? width : height)
        : params[5] == 2 ? float(params[c]) : 1.0f;
      vectors[i] = mix(top, bottom, fy - float(y0)) / divisor;
      """#)

  private static let assess = MLXFast.metalKernel(
    name: "mlxdlss_separable_motion_quality", inputNames: ["current", "previous", "backward", "forward", "params"],
    outputNames: ["confidence", "error"], source: #"""
      uint i = thread_position_in_grid.x, width = params[0], height = params[1];
      if (i >= width * height) return;
      float2 pixels = float2(backward[i * 2] * float(width), backward[i * 2 + 1] * float(height));
      float2 position = float2(i % width, i / width) + pixels;
      if (!all(isfinite(position))) { confidence[i] = 0.0f; error[i] = 1.0f; return; }
      bool inside = position.x >= 0 && position.x <= width - 1 && position.y >= 0 && position.y <= height - 1;
      float2 reverse = float2(
        read_image(forward, position.x, position.y, 0, 2, width, height) * float(width),
        read_image(forward, position.x, position.y, 1, 2, width, height) * float(height));
      float3 warped;
      for (uint c = 0; c < 3; ++c) warped[c] = read_image(previous, position.x, position.y, c, 3, width, height);
      float3 rgb = float3(current[i * 3], current[i * 3 + 1], current[i * 3 + 2]);
      float difference = dot(abs(rgb - warped), float3(1.0f / 3.0f));
      float photo = clamp((0.12f - difference) / 0.09f, 0.0f, 1.0f);
      float tolerance = 0.01f * (dot(pixels, pixels) + dot(reverse, reverse)) + 0.5f;
      confidence[i] = inside && all(isfinite(reverse)) && dot(pixels + reverse, pixels + reverse) <= tolerance ? photo : 0.0f;
      error[i] = abs(dot(rgb - warped, float3(0.2126f, 0.7152f, 0.0722f)));
      """#, header: sampleHeader)

  private static let erode = MLXFast.metalKernel(
    name: "mlxdlss_separable_motion_unused_erode", inputNames: ["input", "params"], outputNames: ["confidence"],
    source: #"""
      uint i = thread_position_in_grid.x; int width = params[0], height = params[1];
      if (i >= uint(width * height)) return;
      int x = i % width, y = i / width;
      bool valid = true;
      for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx)
        valid = valid && input[clamp(y + dy, 0, height - 1) * width + clamp(x + dx, 0, width - 1)] > 0.0f;
      confidence[i] = valid ? input[i] : 0.0f;
      """#)

  private static let resizeKernel = MLXFast.metalKernel(
    name: "mlxdlss_separable_motion_guide_resize", inputNames: ["input", "params"], outputNames: ["output"],
    source: #"""
      uint i = thread_position_in_grid.x, width = params[3], height = params[4], channels = params[2];
      if (i >= width * height * channels) return;
      uint c = i % channels, x = (i / channels) % width, y = i / (channels * width);
      float sx = (float(x) + 0.5f) * float(params[0]) / float(width) - 0.5f;
      float sy = (float(y) + 0.5f) * float(params[1]) / float(height) - 0.5f;
      if (nearest) {
        uint xx = uint(clamp(floor(sx + 0.5f), 0.0f, float(params[0] - 1)));
        uint yy = uint(clamp(floor(sy + 0.5f), 0.0f, float(params[1] - 1)));
        output[i] = input[(yy * params[0] + xx) * channels + c];
      } else {
        output[i] = read_image(input, sx, sy, c, channels, params[0], params[1]);
      }
      """#, header: sampleHeader)
}
