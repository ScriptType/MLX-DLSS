import CoreVideo
import Foundation
import MLX
import XCTest
@testable import DLSSMLX

final class MLXVideoMotionTests: XCTestCase {
  func testTinyConstantBuffersRejectInvalidMotionWithoutPassingNaNToRendering() throws {
    let current = try frame(0.25), previous = try frame(0.25)
    let zero = try flow(0, 0)
    let valid = try MLXVideoMotion(current: current, previous: previous,
      backward: zero, forward: zero, units: .sourcePixels)
    XCTAssertEqual(valid.reliableFraction, 1)
    XCTAssertEqual(valid.warpedLumaError, 0)
    XCTAssertFalse(valid.reset)
    let invalid = try MLXVideoMotion(current: current, previous: previous,
      backward: flow(.nan, .infinity), forward: zero, units: .sourcePixels, sceneCutThreshold: 0)
    XCTAssertEqual(invalid.vectors.asArray(Float.self), [0, 0, 0, 0])
    XCTAssertEqual(invalid.confidence.asArray(Float.self), [0, 0])
    XCTAssertEqual(invalid.reliableFraction, 0)
    XCTAssertFalse(invalid.reset, "Disabling cut detection must still sanitize invalid vectors")
  }

  func testSceneCutAndDisabledCutDetectionUseIndependentLumaEvidence() throws {
    let zero = try flow(0, 0)
    for threshold: Float in [0, 0.3] {
      let cut = try MLXVideoMotion(current: frame(1), previous: frame(0),
        backward: zero, forward: zero, units: .normalizedUV, sceneCutThreshold: threshold)
      XCTAssertEqual(cut.warpedLumaError, 1, accuracy: 1e-6)
      XCTAssertEqual(cut.reliableFraction, 0)
      XCTAssertEqual(cut.reset, threshold > 0)
    }
  }

  private func frame(_ value: Float) throws -> MLXVideoFrame {
    try [Float](repeating: value, count: 6).withUnsafeBytes {
      try MLXVideoFrame(rgb: Data($0), width: 2, height: 1)
    }
  }

  private func flow(_ x: Float, _ y: Float) throws -> MLXPixelBuffer {
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(nil, 1, 1, kCVPixelFormatType_TwoComponent32Float,
      [kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &buffer)
    XCTAssertEqual(status, kCVReturnSuccess)
    let pixels = try XCTUnwrap(buffer)
    CVPixelBufferLockBaseAddress(pixels, [])
    defer { CVPixelBufferUnlockBaseAddress(pixels, []) }
    let values = CVPixelBufferGetBaseAddress(pixels)!.assumingMemoryBound(to: Float.self)
    values[0] = x; values[1] = y
    return MLXPixelBuffer(pixels)
  }
}
