import Foundation
import MLX
import XCTest
@testable import DLSSMLX

final class MLXVideoOutputTests: XCTestCase {
  func testBoxAveragesEveryPixelAndKeepsChannelsSeparate() throws {
    for factor in [2, 3, 4] {
      let h = 3, w = 5, inputWidth = w * factor, inputHeight = h * factor
      let values = (0..<inputWidth * inputHeight * 3).map { Float(($0 * 7) % 113) / 128 }
      let image = MLXArray(values, [1, inputHeight, inputWidth, 3])
      let composition = MLXVideoComposition(options: try .init(width: w, height: h))
      let result = composition.resample(image, width: w, height: h).asArray(Float.self)
      for y in 0..<h { for x in 0..<w { for c in 0..<3 {
        var expected: Double = 0
        for dy in 0..<factor { for dx in 0..<factor {
          expected += Double(values[((y * factor + dy) * inputWidth + x * factor + dx) * 3 + c])
        } }
        XCTAssertEqual(result[(y * w + x) * 3 + c], Float(expected / Double(factor * factor)), accuracy: 1e-7)
      } } }
    }
  }

  func testLanczosMatchesIndependentPillowFloatReferenceAtOddBorders() throws {
    // Pillow mode F, 7x5 -> 4x3, LANCZOS; generated through the Python reference resample().
    let expected: [Float] = [
      0.419277698, 0.698524714, 0.389183372, 0.314154357, 0.385477483, 0.550643325,
      0.431777924, 0.561021507, 0.490442723, 0.324786335, 0.334010839, 0.557896972,
      0.414420158, 0.536334753, 0.658504486, 0.551307559, 0.327879727, 0.448351532,
      0.491240710, 0.386519760, 0.653378010, 0.611716390, 0.418883830, 0.201583683,
      0.713650227, 0.394835740, 0.348353028, 0.358032048, 0.549260914, 0.415807307,
      0.523047745, 0.523963571, 0.355606705, 0.313166648, 0.542938113, 0.517066598,
    ]
    let values: [Float] = (0..<105).map { Float(($0 * 7) % 31) / 32 }
    let image = MLXArray(values, [1, 5, 7, 3])
    let composition = MLXVideoComposition(options: try .init(width: 4, height: 3))
    let result = composition.resample(image, width: 4, height: 3).asArray(Float.self)
    for (actual, reference) in zip(result, expected) { XCTAssertEqual(actual, reference, accuracy: 2e-7) }
    // A new extent invalidates cached axis coefficients, including one-pixel targets.
    for (height, width) in [(1, 1), (3, 7), (5, 4)] {
      let constant = MLXArray.full([1, 5, 7, 3], values: MLXArray(Float(0.375)))
      let resized = composition.resample(constant, width: width, height: height).asArray(Float.self)
      for value in resized { XCTAssertEqual(value, 0.375, accuracy: 2e-7) }
    }
  }

  func testDetailMatchesDirectTwoDimensionalGaussianWithReplicatedEdges() throws {
    for (h, w) in [(1, 1), (3, 5)] {
      let shape = [1, h, w, 3]
      let source = (0..<h * w * 3).map { Float(($0 * 11) % 29) / 32 }
      var display = source.map { $0 + 0.2 }
      display[0] = -0.5
      display[display.count - 1] = 1.5
      for (detail, colour, radius): (Float, Float, Float) in [(0, 0, 0.5), (2, 0.5, 1.25), (0, 2, 4), (1, 1, 4)] {
        let options = try MLXVideoOutputOptions(width: w, height: h, detailStrength: detail, colourStrength: colour, radius: radius)
        let composition = MLXVideoComposition(options: options)
        // Transpose twice through a different backing extent to exercise noncontiguous HWC input.
        let sourceArray = MLXArray(source, shape).transposed(0, 2, 1, 3)
        let displayArray = MLXArray(display, shape).transposed(0, 2, 1, 3)
        let stridedComposition = MLXVideoComposition(options: try .init(width: h, height: w,
          detailStrength: detail, colourStrength: colour, radius: radius))
        let strided = stridedComposition(displayArray, source: sourceArray).transposed(0, 2, 1, 3).asArray(Float.self)
        let result = composition(MLXArray(display, shape), source: MLXArray(source, shape)).asArray(Float.self)
        let extent = Int(ceil(3 * Double(radius)))
        let kernel = (-extent...extent).map { exp(-Double($0 * $0) / (2 * Double(radius) * Double(radius))) }
        let normalizer = pow(kernel.reduce(0, +), 2)
        for y in 0..<h { for x in 0..<w { for c in 0..<3 {
          var low: Double = 0
          for dy in -extent...extent { for dx in -extent...extent {
            let yy = min(h - 1, max(0, y + dy)), xx = min(w - 1, max(0, x + dx))
            let p = (yy * w + xx) * 3 + c
            low += Double(display[p] - source[p]) * kernel[dy + extent] * kernel[dx + extent] / normalizer
          } }
          let p = (y * w + x) * 3 + c
          let combined = Double(source[p]) + Double(colour) * low + Double(detail) * (Double(display[p] - source[p]) - low)
          let expected = detail == 1 && colour == 1 ? display[p] : Float(min(1, max(0, combined)))
          XCTAssertEqual(result[p], expected, accuracy: 5e-7)
          XCTAssertEqual(strided[p], expected, accuracy: 5e-7)
        } } }
      }
    }
  }

  func testFinalPackingClampsAndRoundsWithoutQuantizingFloatFrames() throws {
    let values: [Float] = [-1, 0, 0.5001, 0.5, 1, 2, 0.25, 0.1, 0.9]
    let frame = MLXArray(values, [1, 1, 3, 3])
    for format in [MLXVideoOutputOptions.Format.f32, .u8, .u16] {
      let composition = MLXVideoComposition(options: try .init(width: 3, height: 1, format: format))
      let result = composition.pack(frame).asType(.float32).asArray(Float.self)
      let scale: Float = format == .u8 ? 255 : 65535
      let expected = format == .f32 ? values : values.map { (min(1, max(0, $0)) * scale + 0.5).rounded(.down) }
      XCTAssertEqual(result, expected)
    }
    for radius: Float in [0, -1, .nan, .infinity] {
      XCTAssertThrowsError(try MLXVideoOutputOptions(width: 1, height: 1, radius: radius))
    }
    XCTAssertThrowsError(try MLXVideoOutputOptions(width: Int.max, height: Int.max))
    XCTAssertThrowsError(try MLXVideoFrameGenerationOptions(weightsURL: URL(fileURLWithPath: "/missing"), factor: 1))
  }
}
