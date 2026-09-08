import Foundation
import MLX
import XCTest
@testable import DLSSMLX

final class VideoSuperResolutionTests: XCTestCase {
  private func zeroWeights() -> [String: MLXArray] {
    VideoSuperResolver.weightShapes.mapValues { zeros($0, dtype: .float16) }
  }

  func testOutputHeadQuantizationShufflePaddingAndBatch() throws {
    var weights = zeroWeights()
    let bias = (0..<48).map { Float16(Float($0 - 24) / 24) }
    weights["output.project.bias"] = MLXArray(bias)
    let model = try VideoSuperResolver(weights: weights)
    let h = 9, w = 11
    let pixels = (0..<(2*h*w*3)).map { i in
      Float(i / (h*w*3)) * 0.5 + 0.15 + Float(i % 3) * 0.05
    }
    let output = try model.upscale(MLXArray(pixels, [2,h,w,3]))
    XCTAssertEqual(output.shape, [2,h*2,w*2,3])
    let values = output.asArray(Float.self)
    for n in 0..<2 {
      for y in 0..<(h*2) {
        for x in 0..<(w*2) {
          for c in 0..<3 {
            let color = pixels[n*h*w*3+c]
            let quantized = (color * 255).rounded(.toNearestOrEven) / 255
            let base = Float(Float16((quantized - 0.5) * 2))
            let head = Float(Float16(tanh(Float(bias[c*16+(y%4)*4+x%4]))))
            let composed = Float(Float16(min(max((base + head) * 0.5 + 0.5, 0), 1)))
            let expected = floor(composed * 255) / 255
            let i = ((n*h*2+y)*w*2+x)*3+c
            XCTAssertEqual(values[i], expected, accuracy: 1.01/255)
          }
        }
      }
    }
  }

  func testRejectsInvalidWeightsAndInputShapes() throws {
    var weights = zeroWeights()
    weights.removeValue(forKey: "encoder0.conv0.weight")
    XCTAssertThrowsError(try VideoSuperResolver(weights: weights))
    weights = zeroWeights()
    weights["output.project.bias"] = zeros([47], dtype: .float16)
    XCTAssertThrowsError(try VideoSuperResolver(weights: weights))
    weights = zeroWeights()
    weights["output.project.bias"] = MLXArray.full([48], values: MLXArray(Float.infinity))
    XCTAssertThrowsError(try VideoSuperResolver(weights: weights))
    let model = try VideoSuperResolver(weights: zeroWeights())
    XCTAssertThrowsError(try model.upscale(zeros([1,32,32,4])))
    XCTAssertThrowsError(try model.upscale(zeros([32,32,3])))
    XCTAssertThrowsError(try model.upscale(zeros([1,32,32,3], dtype: .uint8)))
  }

  /// The manifest and vendor captures stay outside the repository.
  func testVendorParityWhenConfigured() throws {
    struct Case: Decodable { let input: String; let output: String; let maximumError: Float? }
    struct Manifest: Decodable { let weights: String; let cases: [Case] }
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_VSR_REFERENCE"] else {
      throw XCTSkip("set MLXDLSS_VSR_REFERENCE to a private oracle manifest")
    }
    let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    XCTAssertFalse(manifest.cases.isEmpty)
    let model = try VideoSuperResolver(weightsURL: URL(fileURLWithPath: manifest.weights))
    for item in manifest.cases {
      let input = try loadArray(url: URL(fileURLWithPath: item.input))
      let reference = try loadArray(url: URL(fileURLWithPath: item.output))
      let result = try model.upscale(input.expandedDimensions(axis: 0)).squeezed(axis: 0)
      XCTAssertEqual(result.shape, reference.shape)
      let difference = abs(result - reference)
      XCTAssertTrue(isFinite(result).all().item(Bool.self))
      XCTAssertLessThanOrEqual(difference.max().item(Float.self), item.maximumError ?? 1.01/255, item.input)
      XCTAssertLessThan(difference.mean().item(Float.self), 0.001, item.input)
    }
  }
}
