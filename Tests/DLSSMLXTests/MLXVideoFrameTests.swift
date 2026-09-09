import CoreVideo
import Foundation
import MLX
import XCTest
@testable import DLSSMLX

final class MLXVideoFrameTests: XCTestCase, @unchecked Sendable {
  func testNativePixelBufferRoundTripPreservesRowsAndChannels() async throws {
    // An odd width forces IOSurface row padding, which must not become pixels.
    let width = 37, height = 19
    let values = (0..<width * height * 3).map { Float(($0 * 41) % 307) / 255 - 0.1 }
    let input = try values.withUnsafeBytes { try MLXVideoFrame(rgb: Data($0), width: width, height: height) }
    for half in [false, true] {
      let writer = try MLXPixelBufferWriter(width: width, height: height, halfOutput: half)
      let buffer = try await writer.write(input)
      XCTAssertGreaterThan(CVPixelBufferGetBytesPerRow(buffer.buffer), width * (half ? 8 : 4))
      let actual = try MLXVideoFrame(pixelBuffer: buffer).array.asArray(Float.self)
      for (a, v) in zip(actual, values) {
        let bounded = max(0, min(1, v))
        let expected = half ? Float(Float16(v)) : floor(bounded * 255 + 0.5) / 255
        XCTAssertEqual(a, expected, accuracy: 1e-7)
      }
    }
  }

  func testPixelBufferImportRejectsNonSurfaceAndWrongShape() throws {
    XCTAssertThrowsError(try MLXVideoFrame(rgb: Data(), width: 0, height: 1))
    XCTAssertThrowsError(try MLXVideoFrame(rgb: Data(), width: 1, height: 1))
    var buffer: CVPixelBuffer?
    XCTAssertEqual(CVPixelBufferCreate(nil, 16, 16, kCVPixelFormatType_32BGRA, nil, &buffer), 0)
    XCTAssertThrowsError(try MLXVideoFrame(pixelBuffer: MLXPixelBuffer(XCTUnwrap(buffer))))
  }
}
