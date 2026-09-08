import XCTest
@testable import DLSSMedia

final class MediaOptionsTests: XCTestCase {
  func testNewNativeJobsUseTemporalAndRejectInvalidRanges() throws {
    var options = MediaProcessingOptions(renderingModel: URL(fileURLWithPath: "/model"))
    XCTAssertTrue(options.temporal)
    try options.validate()
    options.frameLimit = 0
    XCTAssertThrowsError(try options.validate())
    options.frameLimit = nil
    options.processingScale = .nan
    XCTAssertThrowsError(try options.validate())
    XCTAssertThrowsError(try MediaProcessingOptions().validate())
  }

  func testSuperResolutionCanRunAloneButDoesNotEnableSlowMotion() throws {
    var options = MediaProcessingOptions(superResolutionWeights: URL(fileURLWithPath: "/vsr.safetensors"))
    try options.validate()
    options.slowMotion = true
    XCTAssertThrowsError(try options.validate())
  }
}
