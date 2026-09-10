import Foundation
import MLX
import DLSSCore
import XCTest

@testable import DLSSMLX

final class MLXNeuralRenderingDisplayCodecTests: XCTestCase {
  func testDeviceEncodeMatchesPortableReference() throws {
    let original = try tensor(
      pattern(count: 8 * 8 * 3, multiplier: 7, modulus: 41, divisor: 10),
      width: 8,
      height: 8
    )
    let configuration = NeuralRenderingDisplayCodecConfiguration(
      whitePoint: 1.25
    )
    let expected = try NeuralRenderingDisplayCodec.encode(
      original,
      configuration: configuration
    )

    let output = MLXNeuralRenderingDisplayCodec().encode(
      array(original),
      configuration: configuration
    )
    eval(output)

    assertClose(output.asArray(Float.self), values(expected), tolerance: 0.000_002)
  }

  func testDeviceResolveMatchesPortableReference() throws {
    let proxy = try tensor(
      pattern(count: 8 * 8 * 3, multiplier: 3, modulus: 37, divisor: 36),
      width: 8,
      height: 8
    )
    let model = try tensor(
      pattern(
        count: 8 * 8 * 3,
        multiplier: 5,
        addend: 1,
        modulus: 39,
        divisor: 38
      ),
      width: 8,
      height: 8
    )
    let original = try tensor(
      pattern(
        count: 8 * 8 * 3,
        multiplier: 11,
        addend: 2,
        modulus: 53,
        divisor: 9
      ),
      width: 8,
      height: 8
    )
    let configuration = NeuralRenderingDisplayCodecConfiguration(
      whitePoint: 1.5,
      transferStrength: 0.8,
      colorStrength: 0.65,
      maximumLuminanceRatio: 1.75
    )
    let expected = try NeuralRenderingDisplayCodec.resolve(
      proxy: proxy,
      model: model,
      original: original,
      configuration: configuration
    )

    let output = MLXNeuralRenderingDisplayCodec().resolve(
      proxy: array(proxy),
      model: array(model),
      original: array(original),
      configuration: configuration
    )
    eval(output)

    assertClose(output.asArray(Float.self), values(expected), tolerance: 0.000_02)
  }

  func testZeroStrengthReturnsOriginalWithoutADeviceRoundTrip() throws {
    let proxy = MLXArray([Float(0.1), 0.2, 0.3], [1, 1, 1, 3])
    let model = MLXArray([Float(0.8), 0.7, 0.6], [1, 1, 1, 3])
    let original = MLXArray([Float(4), 2, 1], [1, 1, 1, 3])

    let output = MLXNeuralRenderingDisplayCodec().resolve(
      proxy: proxy,
      model: model,
      original: original,
      configuration: .init(transferStrength: 0)
    )
    XCTAssertTrue(output === original)
    eval(output)

    XCTAssertEqual(output.asArray(Float.self), [4, 2, 1])
    let originalFrame = MLXVideoFrame(original)
    let frameOutput = MLXNeuralRenderingDisplayCodec().resolve(
      proxy: MLXVideoFrame(proxy), model: MLXVideoFrame(model), original: originalFrame,
      configuration: .init(transferStrength: 0))
    XCTAssertTrue(frameOutput === originalFrame)
  }

  func testLazyResolvesRetainIndependentEffectParameters() throws {
    let original = try tensor([1000, 1, 0, 0, 800, 20, -0.5, 20, 800, 5, 10, 15], width: 4, height: 1)
    let firstConfiguration = NeuralRenderingDisplayCodecConfiguration(whitePoint: 203,
      transferStrength: 0.25, colorStrength: 0, maximumLuminanceRatio: 1.5, workingPrimaries: .bt2020)
    let secondConfiguration = NeuralRenderingDisplayCodecConfiguration(whitePoint: 203,
      transferStrength: 0.875, colorStrength: 1, maximumLuminanceRatio: 1.5, workingPrimaries: .bt2020)
    let proxy = try NeuralRenderingDisplayCodec.encode(original, configuration: firstConfiguration)
    let model = try tensor(values(proxy).map { min(1, max(0, $0 * 0.8 + 0.04)) }, width: 4, height: 1)
    let originalArray = array(original), proxyArray = array(proxy), modelArray = array(model)
    let codec = MLXNeuralRenderingDisplayCodec()
    let first = codec.resolve(proxy: proxyArray, model: modelArray, original: originalArray,
      configuration: firstConfiguration)
    let second = codec.resolve(proxy: proxyArray, model: modelArray, original: originalArray,
      configuration: secondConfiguration)
    // Build both graphs before evaluating either; the newer call must not
    // overwrite the earlier graph's parameters, including in reverse order.
    eval(second)
    eval(first)
    let firstValues = first.asArray(Float.self), secondValues = second.asArray(Float.self)
    XCTAssertNotEqual(firstValues, secondValues)
    for (actual, configuration) in [(firstValues, firstConfiguration), (secondValues, secondConfiguration)] {
      let expected = try NeuralRenderingDisplayCodec.resolve(proxy: proxy, model: model,
        original: original, configuration: configuration)
      assertClose(actual, values(expected), tolerance: 0.0005)
    }
  }

  func testEffectEndpointsMatchFrozenSpecializedCodec() throws {
    let original = try tensor([1000, 1, 0, 0, 800, 20, -0.5, 20, 800, 0, 0, 0], width: 4, height: 1)
    let proxy = try tensor([0.8, 0.01, 0.2, 0.02, 0.9, 0.1, 0.04045, 0.1, 0.7, 0, 0, 0], width: 4, height: 1)
    let model = try tensor([0.6, 0.03, 0.25, 0.04, 0.75, 0.15, 0.03, 0.2, 0.8, 0, 0, 0], width: 4, height: 1)
    let originalArray = array(original), proxyArray = array(proxy), modelArray = array(model)
    let codec = MLXNeuralRenderingDisplayCodec()
    let baselineCodec = FrozenRuntimeHDRControlsCodec()
    for primaries in [NeuralRenderingDisplayCodecConfiguration.WorkingPrimaries.bt709, .bt2020] {
      for displayReferred in [false, true] {
        for transfer in [Float(0), 1] {
          for colour in [Float(0), 1] {
            let configuration = NeuralRenderingDisplayCodecConfiguration(whitePoint: 203,
              transferStrength: transfer, colorStrength: colour, maximumLuminanceRatio: 1.5,
              inputIsDisplayReferred: displayReferred, workingPrimaries: primaries)
            let actual = codec.resolve(proxy: proxyArray, model: modelArray, original: originalArray,
              configuration: configuration)
            if transfer == 0 { XCTAssertTrue(actual === originalArray) }
            let baseline = baselineCodec.resolve(proxy: proxyArray, model: modelArray,
              original: originalArray, configuration: configuration)
            XCTAssertEqual(actual.asArray(Float.self).map(\.bitPattern),
              baseline.asArray(Float.self).map(\.bitPattern))
          }
        }
      }
    }
  }

  func testBT2020IdentityAndEnhancedWideGamutMatchCPU() throws {
    let original = try tensor([1000, 1, 0, 0, 1000, 0, 1000, 1000, 1000, -0.5, 20, 800], width: 4, height: 1)
    let config = NeuralRenderingDisplayCodecConfiguration(whitePoint: 203, transferStrength: 0.7,
      colorStrength: 0.5, maximumLuminanceRatio: 1.5, workingPrimaries: .bt2020)
    let codec = MLXNeuralRenderingDisplayCodec()
    let proxy = codec.encode(array(original), configuration: config)
    let identity = codec.resolve(proxy: proxy, model: proxy, original: array(original), configuration: config)
    assertClose(identity.asArray(Float.self), values(original), tolerance: 0.0002)
    let hostProxy = try NeuralRenderingDisplayCodec.encode(original, configuration: config)
    assertClose(proxy.asArray(Float.self), values(hostProxy), tolerance: 0.000002)
    let hostModel = try tensor(values(hostProxy).map { min(1, max(0, $0 * 0.9 + 0.02)) }, width: 4, height: 1)
    let expected = try NeuralRenderingDisplayCodec.resolve(proxy: hostProxy, model: hostModel,
      original: original, configuration: config)
    let actual = codec.resolve(proxy: proxy, model: array(hostModel), original: array(original), configuration: config)
    assertClose(actual.asArray(Float.self), values(expected), tolerance: 0.0005)
    XCTAssertGreaterThan(actual.asArray(Float.self).max()!, 203)
  }

  private func tensor(
    _ values: [Float],
    width: Int,
    height: Int
  ) throws -> HostTensor {
    try HostTensor(
      descriptor: TensorDescriptor(
        name: "color",
        shape: [1, height, width, 3],
        dataType: .float32,
        layout: .nhwc
      ),
      bytes: values.withUnsafeBytes { Data($0) }
    )
  }

  private func pattern(
    count: Int,
    multiplier: Int,
    addend: Int = 0,
    modulus: Int,
    divisor: Float
  ) -> [Float] {
    (0..<count).map {
      Float(($0 * multiplier + addend) % modulus) / divisor
    }
  }

  private func array(_ tensor: HostTensor) -> MLXArray {
    MLXArray(tensor.bytes, tensor.descriptor.shape, dtype: .float32)
  }

  private func values(_ tensor: HostTensor) -> [Float] {
    tensor.bytes.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: MemoryLayout<Float>.size).map {
        bytes.loadUnaligned(fromByteOffset: $0, as: Float.self)
      }
    }
  }

  private func assertClose(
    _ actual: [Float],
    _ expected: [Float],
    tolerance: Float,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    let maximum = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    XCTAssertLessThanOrEqual(maximum, tolerance, file: file, line: line)
  }
}
