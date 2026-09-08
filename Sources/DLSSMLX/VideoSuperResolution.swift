import Foundation
import MLX
import MLXNN

/// Experimental RTX Video Super Resolution: High Bitrate Low (mode 16), 2x.
/// Uses independently supplied logical weights from VSR 1.8.2. Each frame is
/// independent; input and output follow the vendor's 8-bit RGB quantization.
public final class VideoSuperResolver {
  public struct Error: Swift.Error, CustomStringConvertible, LocalizedError {
    public let description: String
    public var errorDescription: String? { description }
  }

  public static let modelIdentifier = "rtx-vsr-1.8.2-highbitrate-low-2x"

  /// Logical weight shapes are [output channels, input channels, height, width].
  public static let weightShapes: [String: [Int]] = {
    var shapes: [String: [Int]] = [:]
    func convolution(_ name: String, _ input: Int, _ output: Int, _ size: Int = 3) {
      shapes[name + ".weight"] = [output, input, size, size]
      shapes[name + ".bias"] = [output]
    }
    func block(_ name: String, _ input: Int, _ output: Int) {
      convolution(name + ".conv0", input, output)
      convolution(name + ".conv1", output, output)
      convolution(name + ".shortcut", input, output, 1)
    }
    for (index, channels) in [(16,16),(16,32),(32,32),(32,64),(64,64)].enumerated() {
      block("encoder\(index)", channels.0, channels.1)
    }
    for (index, channels) in [(64,64),(64,32),(32,32),(32,16)].enumerated() {
      convolution("decoder\(index).upsample", channels.0, channels.1)
      block("decoder\(index)", channels.1 * 2, channels.1)
    }
    convolution("output.conv", 16, 64)
    convolution("output.project", 64, 48, 1)
    return shapes
  }()

  private struct Layer {
    let weight: MLXArray
    let bias: MLXArray
    func callAsFunction(_ input: MLXArray) -> MLXArray {
      conv2d(input, weight, padding: IntOrPair(weight.dim(1) / 2)) + bias
    }
  }
  private let layers: [String: Layer]
  private let nearest = Upsample(scaleFactor: 2.0, mode: .nearest)
  private let cubic = Upsample(scaleFactor: 2.0, mode: .cubic())

  public convenience init(weightsURL: URL) throws {
    try self.init(weights: loadArrays(url: weightsURL, stream: .cpu))
  }

  public init(weights: [String: MLXArray]) throws {
    let expected = Set(Self.weightShapes.keys)
    guard Set(weights.keys) == expected else {
      let missing = expected.subtracting(weights.keys).sorted()
      let extra = Set(weights.keys).subtracting(expected).sorted()
      throw Error(description: "VSR weights: missing \(missing), unexpected \(extra)")
    }
    var prepared: [String: MLXArray] = [:]
    for (name, shape) in Self.weightShapes {
      let source = weights[name]!
      guard source.shape == shape, source.dtype == .float16 || source.dtype == .float32 else {
        throw Error(description: "\(name): expected floating-point weights of shape \(shape), got \(source.shape) \(source.dtype)")
      }
      let value = source.asType(.float16)
      guard isFinite(value).all().item(Bool.self) else {
        throw Error(description: "\(name): weights must be finite in float16")
      }
      prepared[name] = value
    }
    var layers: [String: Layer] = [:]
    for name in expected where name.hasSuffix(".weight") {
      let stem = String(name.dropLast(7))
      layers[stem] = Layer(weight: prepared[name]!.transposed(0,2,3,1), bias: prepared[stem + ".bias"]!)
    }
    self.layers = layers
  }

  private func activation(_ input: MLXArray) -> MLXArray {
    let value = input.asType(.float32)
    return MLX.which(value .>= 0, value, exp(value) - 1).asType(.float16)
  }

  private func block(_ input: MLXArray, _ name: String) -> MLXArray {
    let first = activation(layers[name + ".conv0"]!(input))
    let second = activation(layers[name + ".conv1"]!(first))
    return second + layers[name + ".shortcut"]!(input)
  }

  /// Upscale [N,H,W,3] floating-point RGB in [0,1] to [N,2H,2W,3].
  /// Values are clamped and rounded to RGB8 before inference. Padding repeats
  /// border pixels to a multiple of 32 and is removed from the result.
  public func upscale(_ rgb: MLXArray) throws -> MLXArray {
    guard rgb.ndim == 4, rgb.dim(0) > 0, rgb.dim(1) > 0, rgb.dim(2) > 0, rgb.dim(3) == 3,
      rgb.dtype == .float16 || rgb.dtype == .float32
    else { throw Error(description: "VSR expects nonempty [N,H,W,3] floating-point RGB") }
    let n = rgb.dim(0), originalH = rgb.dim(1), originalW = rgb.dim(2)
    let h = (originalH + 31) / 32 * 32, w = (originalW + 31) / 32 * 32
    let quantized = round(clip(rgb.asType(.float32), min: 0, max: 1) * 255) / 255
    var base = ((quantized - 0.5) * 2).asType(.float16)
    if h > originalH {
      let edge = base[0..., (originalH-1)..<originalH, 0..., 0...]
      base = concatenated([base, broadcast(edge, to: [n,h-originalH,originalW,3])], axis: 1)
    }
    if w > originalW {
      let edge = base[0..., 0..., (originalW-1)..<originalW, 0...]
      base = concatenated([base, broadcast(edge, to: [n,h,w-originalW,3])], axis: 2)
    }
    let rgba = concatenated([base, zeros([n,h,w,1], dtype: .float16)], axis: 3)
    var x = rgba.reshaped(n,h/2,2,w/2,2,4).transposed(0,1,3,2,4,5).reshaped(n,h/2,w/2,16)
    var skips: [MLXArray] = []
    for index in 0..<4 {
      x = block(x, "encoder\(index)")
      skips.append(x)
      x = x.asType(.float32).reshaped(n,x.dim(1)/2,2,x.dim(2)/2,2,x.dim(3))
        .mean(axes: [2,4]).asType(.float16)
    }
    x = block(x, "encoder4")
    for index in 0..<4 {
      x = activation(layers["decoder\(index).upsample"]!(nearest(x)))
      x = block(concatenated([x, skips.removeLast()], axis: 3), "decoder\(index)")
    }
    x = layers["output.project"]!(activation(layers["output.conv"]!(x)))
    // The bounded output head is significant on highlights and sharp edges.
    x = tanh(x.asType(.float32)).asType(.float16)
    let residual = x.reshaped(n,h/2,w/2,3,4,4).transposed(0,1,4,2,5,3).reshaped(n,h*2,w*2,3)
    let image = cubic(base.asType(.float32))
    let output = clip((image + residual.asType(.float32)) * 0.5 + 0.5, min: 0, max: 1)
      .asType(.float16).asType(.float32)
    return (floor(output * 255) / 255)[0..., 0..<(originalH*2), 0..<(originalW*2), 0...]
  }
}

/// Completes the lazy graph before passing a frame across actor boundaries.
public actor MLXNativeSuperResolver {
  private let model: VideoSuperResolver

  public init(weightsURL: URL) throws {
    model = try VideoSuperResolver(weightsURL: weightsURL)
  }

  public func upscale(_ frame: MLXVideoFrame) throws -> MLXVideoFrame {
    try MLXVideoFrame(model.upscale(frame.array))
  }
}
