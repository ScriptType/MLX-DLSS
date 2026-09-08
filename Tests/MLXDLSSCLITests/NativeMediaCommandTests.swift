import DLSSCore
import XCTest
@testable import mlxdlss

final class NativeMediaCommandTests: XCTestCase {
  func testNativeVideoDefaultsAndStrictOptions() throws {
    let base = ["input.mp4", "--model", "model.dlssmodel", "--output", "result.mp4"]
    let parsed = try ProcessMediaCommand.parse(arguments: base, video: true)
    XCTAssertTrue(parsed.options.temporal)
    XCTAssertEqual(parsed.options.profile, .standard)
    for invalid in [["--temporal", "maybe"], ["--frames", "0"], ["--motion", "unknown"],
      ["--processing-scale", "nan"], ["--output", "duplicate.mp4"], ["--frames"]] {
      XCTAssertThrowsError(try ProcessMediaCommand.parse(arguments: base + invalid, video: true))
    }
    XCTAssertThrowsError(try ProcessMediaCommand.parse(arguments: ["input.mp4", "--output", "result.mp4"], video: true))
  }

  func testImagesUseNaturalProfileAndRejectVideoControls() throws {
    let base = ["input.png", "--model", "model.dlssmodel", "--output", "result.png"]
    let parsed = try ProcessMediaCommand.parse(arguments: base, video: false)
    XCTAssertEqual(parsed.options.profile, .natural)
    XCTAssertThrowsError(try ProcessMediaCommand.parse(arguments: base + ["--factor", "4"], video: false))
  }
}
