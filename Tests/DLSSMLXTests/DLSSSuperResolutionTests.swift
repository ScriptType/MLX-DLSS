import Foundation
import MLX
import XCTest

@testable import DLSSMLX

final class DLSSSuperResolutionTests: XCTestCase {
  func testGeometryAndWindowCycle() {
    XCTAssertEqual(
      DLSSSuperResolver.geometry(width: 97, height: 65), [97, 65, 194, 130, 256, 256, 208, 144])
    XCTAssertEqual(
      DLSSSuperResolver.geometry(width: 1920, height: 1080),
      [1920, 1080, 3840, 2160, 960, 544, 3840, 2160])
    let expected = [(0, 2), (5, 6), (4, 0), (1, 4), (7, 5), (2, 1), (3, 7), (6, 3)]
    for frame in 0..<128 {
      let specs = DLSSSuperResolver.specifications(width: 960, height: 544, frame: frame)
      XCTAssertEqual(specs.count, 11)
      XCTAssertEqual(specs[5].offsetX, expected[frame % 8].0)
      XCTAssertEqual(specs[5].offsetY, expected[frame % 8].1)
      XCTAssertEqual(specs[0].offsetX, 4)
      XCTAssertEqual(specs[0].offsetY, 4)
      for index in 0..<5 {
        XCTAssertEqual(specs[index].offsetX, specs[10 - index].offsetX)
        XCTAssertEqual(specs[index].offsetY, specs[10 - index].offsetY)
      }
    }
  }

  func testRejectsUnsupportedPackageBeforeLoadingShader() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let manifest = [
      "format": "mlxdlss-sr-v1", "model": "unsupported", "postprocessSHA256": "invalid",
    ]
    try JSONSerialization.data(withJSONObject: manifest).write(
      to: directory.appendingPathComponent("manifest.json"))
    XCTAssertThrowsError(try DLSSSuperResolver(packageURL: directory)) { error in
      XCTAssertTrue(error.localizedDescription.contains("Unsupported DLSS SR package"))
    }
  }

  /// Original library outputs and weights are supplied locally and never committed.
  func testOriginalSequenceAndResetWhenConfigured() throws {
    struct Frame: Decodable {
      let input: String
      let motion: String
      let output: String
    }
    struct Reference: Decodable {
      let model: String
      let width: Int
      let height: Int
      let frames: [Frame]
    }
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_SR_REFERENCE"] else {
      throw XCTSkip("Set MLXDLSS_SR_REFERENCE to a private original-library sequence")
    }
    let reference = try JSONDecoder().decode(
      Reference.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    XCTAssertGreaterThan(reference.frames.count, 1)
    let model = try DLSSSuperResolver(packageURL: URL(fileURLWithPath: reference.model))
    let h = reference.height
    let w = reference.width
    func tensor(_ path: String, _ shape: [Int]) throws -> MLXArray {
      let data = try Data(contentsOf: URL(fileURLWithPath: path))
      XCTAssertEqual(data.count, shape.reduce(4, *))
      return MLXArray(data, shape, dtype: .float32)
    }
    var first: MLXArray?
    for frame in reference.frames {
      let input = try tensor(frame.input, [1, h, w, 3])
      let flow = try tensor(frame.motion, [1, h, w, 2])
      let output = try model.upscale(input, motion: flow)
      let expected = try tensor(frame.output, [1, h * 2, w * 2, 3])
      XCTAssertEqual(output.shape, expected.shape)
      XCTAssertTrue(isFinite(output).all().item(Bool.self))
      let error = abs(output - expected)
      XCTAssertLessThan(error.mean().item(Float.self), 0.00025)
      XCTAssertLessThan((error * error).mean().item(Float.self), 0.0000016)
      if first == nil { first = output }
    }
    model.reset()
    let input = try tensor(reference.frames[0].input, [1, h, w, 3])
    let replay = try model.upscale(input)
    XCTAssertEqual(abs(replay - first!).max().item(Float.self), 0)
    XCTAssertThrowsError(try model.upscale(zeros([1, h, w, 4])))
    XCTAssertThrowsError(try model.upscale(input, motion: zeros([1, h, w, 3])))
    XCTAssertThrowsError(
      try model.upscale(MLXArray.full([1, h, w, 3], values: MLXArray(Float.nan))))
    let odd = try model.upscale(zeros([1, 65, 97, 3]))
    XCTAssertEqual(odd.shape, [1, 130, 194, 3])
    XCTAssertTrue(isFinite(odd).all().item(Bool.self))
    XCTAssertLessThan(abs(odd).max().item(Float.self), 0.00001)
    let constant = try model.upscale(ones([1, 65, 97, 3]) * 0.4, reset: true)
    XCTAssertLessThan(abs(constant - 0.4).max().item(Float.self), 0.003)
  }
}
