import Foundation
import MLX
import XCTest
@testable import DLSSMLX

final class MLXNeuralRenderingInputRangeTests: XCTestCase {
  func testBoundedAdmissionAfterBothResizePassesMatchesIndependentPillowReference() throws {
    // Pillow mode F, 48x27 -> 8x5 LANCZOS. R/G are binary vertical/horizontal
    // steps; B is their XOR, exercising both passes before the single clamp.
    let red: [Float] = [0, 0.001341157593, 0.002226327546, -0.01602915861,
      0.8138291836, 1.04183495, 0.990865469, 1]
    let green: [Float] = [0.01708492823, -0.0694681555, 0.406299144, 1.069468141, 0.9829150438]
    let blue: [Float] = [
      0.01708492823, 0.0183802601, 0.01923518255, 0.001603483805, 0.8031057119, 1.023320556, 0.9740926623, 0.9829150438,
      -0.0694681555, -0.06794065982, -0.06693250686, -0.08772434294, 0.8574314713, 1.117115498, 1.059064507, 1.069468141,
      0.406299144, 0.4065504968, 0.4067163765, 0.4032952487, 0.5588121414, 0.6015408039, 0.5919889808, 0.5937008262,
      1.069468141, 1.067940712, 1.06693244, 1.087724328, 0.1425685585, -0.1171154678, -0.05906452239, -0.0694681555,
      0.9829150438, 0.9816197157, 0.9807648063, 0.9983964562, 0.1968943328, -0.02332050167, 0.02590731904, 0.01708492823,
    ]
    var pixels = [Float]()
    for y in 0..<27 { for x in 0..<48 {
      let r: Float = x >= 25 ? 1 : 0, g: Float = y >= 14 ? 1 : 0
      pixels += [r, g, r == g ? 0 : 1]
    } }
    let source = MLXArray(pixels, [1, 27, 48, 3])
    let composition = MLXVideoComposition(options: try .init(width: 48, height: 27))
    let bounded = MLXNeuralRenderingDeviceTemporalBackend.prepareVideoInput(source,
      composition: composition, width: 8, height: 5, range: .boundedSRGB)
    let preserved = MLXNeuralRenderingDeviceTemporalBackend.prepareVideoInput(source,
      composition: composition, width: 8, height: 5, range: .preserve)
    let defaultRange = MLXNeuralRenderingDeviceTemporalBackend.prepareVideoInput(source,
      composition: composition, width: 8, height: 5)
    eval(bounded, preserved, defaultRange)
    let actual = bounded.asArray(Float.self), raw = preserved.asArray(Float.self)
    XCTAssertEqual(defaultRange.asArray(Float.self), raw)
    XCTAssertTrue(raw.contains { $0 < -0.01 })
    XCTAssertTrue(raw.contains { $0 > 1.01 })
    XCTAssertTrue(actual.allSatisfy { $0.isFinite && (0...1).contains($0) })
    for y in 0..<5 { for x in 0..<8 {
      for (channel, expected) in [red[x], green[y], blue[y * 8 + x]].enumerated() {
        let index = (y * 8 + x) * 3 + channel
        XCTAssertEqual(raw[index], expected, accuracy: 0.000_001)
        XCTAssertEqual(actual[index], min(1, max(0, expected)), accuracy: 0.000_001)
      }
    } }
    XCTAssertEqual(source.asArray(Float.self), pixels, "Admission must not mutate its source")
  }

  func testAsymmetricUpsizeBoundsOnlyAdmittedProxy() throws {
    // Independent Pillow mode-F LANCZOS reference for 3x2 -> 7x5.
    let red: [Float] = [-0.1458448917, 0.09808924794, 0.5206261277, 1, 1.135776043, 1.033331752, 0.9586853981]
    let green: [Float] = [-0.2027456909, 0.08721781522, 0.5, 0.9127821922, 1.202745676]
    let pixels: [Float] = [0, 0, 0.375, 1, 0, 0.375, 1, 0, 0.375,
                           0, 1, 0.375, 1, 1, 0.375, 1, 1, 0.375]
    let source = MLXArray(pixels, [1, 2, 3, 3])
    let composition = MLXVideoComposition(options: try .init(width: 3, height: 2))
    let raw = MLXNeuralRenderingDeviceTemporalBackend.prepareVideoInput(source,
      composition: composition, width: 7, height: 5).asArray(Float.self)
    let bounded = MLXNeuralRenderingDeviceTemporalBackend.prepareVideoInput(source,
      composition: composition, width: 7, height: 5, range: .boundedSRGB).asArray(Float.self)
    XCTAssertTrue(raw.contains { $0 < -0.01 })
    XCTAssertTrue(raw.contains { $0 > 1.01 })
    XCTAssertTrue(bounded.allSatisfy { $0.isFinite && (0...1).contains($0) })
    for y in 0..<5 { for x in 0..<7 {
      for (channel, expected) in [red[x], green[y], 0.375].enumerated() {
        let index = (y * 7 + x) * 3 + channel
        XCTAssertEqual(raw[index], expected, accuracy: 0.000_001)
        XCTAssertEqual(bounded[index], min(1, max(0, expected)), accuracy: 0.000_001)
      }
    } }
    XCTAssertEqual(source.asArray(Float.self), pixels)
  }

  func testEqualDimensionsPreserveBoundedInputExactlyAndDefaultKeepsExtendedValues() throws {
    let values: [Float] = [0, 1, 0.125, 0.5, 0.375, 0.999]
    let composition = MLXVideoComposition(options: try .init(width: 2, height: 1))
    let input = MLXArray(values, [1, 1, 2, 3])
    let bounded = MLXNeuralRenderingDeviceTemporalBackend.prepareVideoInput(input,
      composition: composition, width: 2, height: 1, range: .boundedSRGB)
    XCTAssertEqual(bounded.asArray(Float.self), values)
    let extended: [Float] = [-0.25, 1.25, 0.5, -2, 203, 1000]
    let original = MLXArray(extended, [1, 1, 2, 3])
    let unchanged = MLXNeuralRenderingDeviceTemporalBackend.prepareVideoInput(original,
      composition: composition, width: 2, height: 1)
    let admitted = MLXNeuralRenderingDeviceTemporalBackend.prepareVideoInput(original,
      composition: composition, width: 2, height: 1, range: .boundedSRGB)
    eval(unchanged, admitted)
    XCTAssertEqual(unchanged.asArray(Float.self), extended)
    XCTAssertEqual(admitted.asArray(Float.self), [0, 1, 0.5, 0, 1, 1])
    XCTAssertEqual(original.asArray(Float.self), extended)
  }
}
