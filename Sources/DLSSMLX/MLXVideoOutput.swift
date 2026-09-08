import Foundation
import DLSSCore
import MLX

/// Display composition is separate from the full-resolution temporal history.
public struct MLXVideoOutputOptions: Equatable, Sendable {
  public enum Format: String, Sendable { case f32, u8, u16 }
  public let width: Int
  public let height: Int
  public let detailStrength: Float
  public let colourStrength: Float
  public let radius: Float
  public let format: Format

  public init(width: Int, height: Int, detailStrength: Float = 1,
              colourStrength: Float = 1, radius: Float = 4, format: Format = .f32) throws {
    guard width > 0, height > 0, height <= Int(Int32.max) / 3,
      width <= Int(Int32.max) / height / 3,
      detailStrength.isFinite, colourStrength.isFinite, radius.isFinite,
      radius > 0, radius < Float(Int32.max / 6)
    else { throw NeuralRenderingDetailComposition.Error.invalidStrength("invalid video output extent, strengths or radius") }
    self.width = width
    self.height = height
    self.detailStrength = detailStrength
    self.colourStrength = colourStrength
    self.radius = radius
    self.format = format
  }
}

public struct MLXVideoFrameGenerationOptions: Sendable {
  public let weightsURL: URL
  public let precision: FrameGenerator.Precision
  public let factor: Int
  public let batch: Int

  public init(weightsURL: URL, precision: FrameGenerator.Precision = .float16,
              factor: Int = 2, batch: Int = 4) throws {
    guard factor >= 2, batch >= 1, factor <= Int(Int32.max) / batch else {
      throw FrameGenerator.Error(description: "frame generation requires factor >= 2 and batch >= 1")
    }
    self.weightsURL = weightsURL
    self.precision = precision
    self.factor = factor
    self.batch = batch
  }
}

/// Owned and called only by the rendering actor. No MLX arrays cross its output boundary.
final class MLXVideoOutput {
  let options: MLXVideoOutputOptions
  private let composition: MLXVideoComposition
  private let generator: FrameGenerator?
  private let phases: [Float]
  private let batch: Int
  private var window: [MLXArray] = []

  init(options: MLXVideoOutputOptions, frameGeneration: MLXVideoFrameGenerationOptions?) throws {
    self.options = options
    composition = MLXVideoComposition(options: options)
    generator = try frameGeneration.map { try FrameGenerator(weightsURL: $0.weightsURL, precision: $0.precision) }
    phases = frameGeneration.map { fg in (1..<fg.factor).map { Float($0) / Float(fg.factor) } } ?? []
    batch = frameGeneration?.batch ?? 1
  }

  func push(_ rendered: MLXArray, source: MLXArray, to output: FileHandle) throws -> Int {
    let display = composition(rendered, source: source)
    guard generator != nil else {
      try write(display, to: output)
      return 1
    }
    let first = window.isEmpty
    window.append(stopGradient(display))
    if first {
      try write(display, to: output)
      return 1
    }
    // Finish this frame's graph before retaining it for the next FG window.
    eval(display)
    return window.count == batch + 1 ? try finish(to: output) : 0
  }

  /// Flush a complete or short final window, retaining only its last frame.
  func finish(to output: FileHandle) throws -> Int {
    guard let generator, window.count > 1 else { return 0 }
    let pairs = window.count - 1
    let a = concatenated(window.dropLast().flatMap { Array(repeating: $0, count: phases.count) }, axis: 0)
    let b = concatenated(window.dropFirst().flatMap { Array(repeating: $0, count: phases.count) }, axis: 0)
    let generated = try generator.interpolationGraph(a, b, phases: (0..<pairs).flatMap { _ in phases })
    // Pack the whole generated batch once; retain float32 originals for subsequent FG inputs.
    let packed = composition.pack(generated)
    eval(packed)
    for pair in 0..<pairs {
      let frames = packed[(pair * phases.count)..<((pair + 1) * phases.count)]
      try writeStorage(frames, to: output)
      try write(window[pair + 1], to: output)
    }
    window = [window[window.count - 1]]
    return pairs * (phases.count + 1)
  }

  private func write(_ frame: MLXArray, to output: FileHandle) throws {
    try writeStorage(composition.pack(frame), to: output)
  }

  private func writeStorage(_ frame: MLXArray, to output: FileHandle) throws {
    let storage = contiguous(frame)
    // Borrowed Data does not retain its MLXArray. The synchronous write must finish first.
    try withExtendedLifetime(storage) {
      try output.write(contentsOf: storage.asData(access: .noCopyIfContiguous).data)
    }
  }
}

/// Float32 NHWC kernels matching the converter's box/Lanczos and Gaussian detail recipe.
/// Coefficients are prepared once per extent; all per-frame pixels stay on the GPU.
final class MLXVideoComposition {
  private let options: MLXVideoOutputOptions
  private let gaussian: MLXArray
  private let strengths: MLXArray
  private var resizePlans: [Int: (input: Int, output: Int, bounds: MLXArray, weights: MLXArray)] = [:]

  init(options: MLXVideoOutputOptions) {
    self.options = options
    let extent = Int(ceil(3 * options.radius))
    let weights = (-extent...extent).map { x in exp(-Float(x * x) / (2 * options.radius * options.radius)) }
    let sum = weights.reduce(Float(0), +)
    gaussian = MLXArray(weights.map { $0 / sum })
    strengths = MLXArray([options.detailStrength, options.colourStrength])
  }

  func callAsFunction(_ rendered: MLXArray, source: MLXArray) -> MLXArray {
    precondition(source.shape == [1, options.height, options.width, 3])
    let display = resample(rendered, width: options.width, height: options.height)
    if options.detailStrength == 1, options.colourStrength == 1 { return display }
    let horizontal = Self.blurHorizontal(
      [display, source, gaussian], grid: (source.size, 1, 1), threadGroup: (256, 1, 1),
      outputShapes: [source.shape], outputDTypes: [.float32])[0]
    return Self.blurCompose(
      [horizontal, display, source, gaussian, strengths], grid: (source.size, 1, 1), threadGroup: (256, 1, 1),
      outputShapes: [source.shape], outputDTypes: [.float32])[0]
  }

  func pack(_ frame: MLXArray) -> MLXArray {
    guard options.format != .f32 else { return frame }
    return Self.quantize([frame], template: [("rgb16", options.format == .u16)],
      grid: (frame.size, 1, 1), threadGroup: (256, 1, 1),
      outputShapes: [frame.shape], outputDTypes: [options.format == .u8 ? .uint8 : .uint16])[0]
  }

  func resample(_ frame: MLXArray, width: Int, height: Int) -> MLXArray {
    let h = frame.shape[1], w = frame.shape[2]
    if w == width, h == height { return frame }
    let shape = [1, height, width, 3]
    if w % width == 0, h % height == 0, w / width == h / height, w / width > 1 {
      return Self.box([frame], template: [("factor", w / width)],
        grid: (width * height * 3, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [shape], outputDTypes: [.float32])[0]
    }
    var result = frame
    for (axis, extent) in [(2, width), (1, height)] where result.shape[axis] != extent {
      let inputExtent = result.shape[axis]
      // One plan per axis: this object processes a fixed video resolution.
      let plan: (bounds: MLXArray, weights: MLXArray)
      if let cached = resizePlans[axis], cached.input == inputExtent, cached.output == extent {
        plan = (cached.bounds, cached.weights)
      } else {
        plan = Self.lanczosPlan(input: inputExtent, output: extent)
        resizePlans[axis] = (inputExtent, extent, plan.bounds, plan.weights)
      }
      var resizedShape = result.shape
      resizedShape[axis] = extent
      result = Self.lanczos([result, plan.bounds, plan.weights], template: [("horizontal", axis == 2)],
        grid: (resizedShape.reduce(1, *), 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [resizedShape], outputDTypes: [.float32])[0]
    }
    return result
  }

  private static func lanczosPlan(input: Int, output: Int) -> (bounds: MLXArray, weights: MLXArray) {
    let scale = Double(input) / Double(output), filterScale = max(1, Double(input) / Double(output))
    let support = 3 * filterScale, taps = 2 * Int(ceil(support)) + 1
    var bounds: [Int32] = [], weights = [Float](repeating: 0, count: output * taps)
    func sinc(_ x: Double) -> Double { x == 0 ? 1 : sin(.pi * x) / (.pi * x) }
    for i in 0..<output {
      let center = (Double(i) + 0.5) * scale
      let start = max(0, Int(center - support + 0.5)), end = min(input, Int(center + support + 0.5))
      let values = (start..<end).map { j -> Double in
        let x = (Double(j) - center + 0.5) / filterScale
        return x >= -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0
      }
      let total = values.reduce(0, +)
      bounds += [Int32(start), Int32(end - start)]
      for (j, value) in values.enumerated() { weights[i * taps + j] = Float(value / total) }
    }
    return (MLXArray(bounds, [output, 2]), MLXArray(weights, [output, taps]))
  }

  private static let header = "#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n"
  private static let box = MLXFast.metalKernel(name: "mlxdlss_video_box", inputNames: ["image"], outputNames: ["output"], source: #"""
    uint i = thread_position_in_grid.x;
    uint width = uint(image_shape[2]) / factor, height = uint(image_shape[1]) / factor;
    if (i >= width * height * 3) return;
    uint c = i % 3, x = (i / 3) % width, y = i / (3 * width);
    float total = 0.0f;
    for (uint dy = 0; dy < factor; ++dy)
      for (uint dx = 0; dx < factor; ++dx)
        total += image[((y * factor + dy) * uint(image_shape[2]) + x * factor + dx) * 3 + c];
    output[i] = total / float(factor * factor);
    """#, header: header)

  private static let lanczos = MLXFast.metalKernel(name: "mlxdlss_video_lanczos", inputNames: ["image", "bounds", "weights"], outputNames: ["output"], source: #"""
    uint i = thread_position_in_grid.x;
    uint width = horizontal ? uint(bounds_shape[0]) : uint(image_shape[2]);
    uint height = horizontal ? uint(image_shape[1]) : uint(bounds_shape[0]);
    if (i >= width * height * 3) return;
    uint c = i % 3, x = (i / 3) % width, y = i / (3 * width), p = horizontal ? x : y;
    int start = bounds[p * 2], count = bounds[p * 2 + 1];
    float total = 0.0f;
    for (int j = 0; j < count; ++j) {
      uint sx = horizontal ? uint(start + j) : x, sy = horizontal ? y : uint(start + j);
      total += image[(sy * uint(image_shape[2]) + sx) * 3 + c] * weights[p * uint(weights_shape[1]) + j];
    }
    output[i] = total;
    """#, header: header)

  private static let blurHorizontal = MLXFast.metalKernel(name: "mlxdlss_video_detail_horizontal", inputNames: ["display", "source", "weights"], outputNames: ["output"], source: #"""
    uint i = thread_position_in_grid.x;
    int width = source_shape[2], height = source_shape[1];
    if (i >= uint(width * height * 3)) return;
    int c = i % 3, x = (i / 3) % width, row = int(i / (3 * width)) * width;
    int taps = weights_shape[0], extent = taps / 2;
    float total = 0.0f;
    for (int j = 0; j < taps; ++j) {
      int s = (row + clamp(x + j - extent, 0, width - 1)) * 3 + c;
      total += (display[s] - source[s]) * weights[j];
    }
    output[i] = total;
    """#, header: header)

  private static let blurCompose = MLXFast.metalKernel(name: "mlxdlss_video_detail_compose", inputNames: ["horizontal", "display", "source", "weights", "strengths"], outputNames: ["output"], source: #"""
    uint i = thread_position_in_grid.x;
    int width = source_shape[2], height = source_shape[1];
    if (i >= uint(width * height * 3)) return;
    int column = i % (3 * width), y = i / (3 * width), taps = weights_shape[0], extent = taps / 2;
    float low = 0.0f;
    for (int j = 0; j < taps; ++j)
      low += horizontal[clamp(y + j - extent, 0, height - 1) * width * 3 + column] * weights[j];
    float change = display[i] - source[i];
    output[i] = clamp(source[i] + strengths[1] * low + strengths[0] * (change - low), 0.0f, 1.0f);
    """#, header: header)

  private static let quantize = MLXFast.metalKernel(name: "mlxdlss_video_quantize", inputNames: ["image"], outputNames: ["output"], source: #"""
    uint i = thread_position_in_grid.x;
    if (i >= uint(image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3])) return;
    output[i] = clamp(image[i], 0.0f, 1.0f) * (rgb16 ? 65535.0f : 255.0f) + 0.5f;
    """#, header: header)
}
