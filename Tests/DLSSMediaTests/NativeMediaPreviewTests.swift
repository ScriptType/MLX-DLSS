import AVFoundation
import DLSSMLX
import Foundation
import XCTest
@testable import DLSSMedia

final class NativeMediaPreviewTests: XCTestCase, @unchecked Sendable {
  func testVideoPreviewSeeksWithoutResizingAndStartsAtFirstFrame() async throws {
    guard #available(macOS 26.0, *) else { throw XCTSkip("Native preview requires macOS 26") }
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let video = directory.appendingPathComponent("timeline.mp4")
    let writer = try NativeVideoWriter(url: video, width: 96, height: 64, frameRate: 10,
      options: MediaProcessingOptions(), hasAudio: false)
    for index in 0..<8 {
      let pixels = [Float](repeating: Float(index) / 10, count: 96 * 64 * 3)
      let frame = try pixels.withUnsafeBytes { try MLXVideoFrame(rgb: Data($0), width: 96, height: 64) }
      try await writer.append(frame, at: CMTime(value: Int64(index), timescale: 10))
    }
    try await writer.finish(at: CMTime(value: 8, timescale: 10))
    let session = try NativeMediaPreview()
    let options = MediaProcessingOptions()
    let first = try await session.render(.init(input: video, isVideo: true, options: options))
    let middle = try await session.render(.init(input: video, isVideo: true, time: 0.4, options: options))
    let last = try await session.render(.init(input: video, isVideo: true, time: 1, options: options))
    XCTAssertEqual(first.time, 0, accuracy: 1e-5)
    XCTAssertEqual(middle.time, 0.4, accuracy: 1e-5)
    XCTAssertEqual(last.time, 0.7, accuracy: 1e-5)
    XCTAssertEqual(middle.duration, 0.8, accuracy: 1e-5)
    XCTAssertEqual(middle.processed.width, 96)
    XCTAssertEqual(middle.processed.height, 64)
    XCTAssertNotEqual(bytes(first.processed), bytes(middle.processed))
    XCTAssertNotEqual(bytes(middle.processed), bytes(last.processed))
    XCTAssertEqual(bytes(middle.original), bytes(middle.processed))
    do {
      _ = try await session.render(.init(input: video, isVideo: true, time: .nan, options: options))
      XCTFail("Invalid seek time must be rejected")
    } catch { XCTAssertTrue(error.localizedDescription.contains("time")) }
  }

  func testLiveControlsMatchImageExportAndDoNotLeakHistory() async throws {
    guard #available(macOS 26.0, *) else { throw XCTSkip("Native preview requires macOS 26") }
    guard let package = ProcessInfo.processInfo.environment["MLXDLSS_NEURAL_RENDERING_PACKAGE"] else {
      throw XCTSkip("Set MLXDLSS_NEURAL_RENDERING_PACKAGE to validate live rendering with real weights")
    }
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let input = directory.appendingPathComponent("input.png"), output = directory.appendingPathComponent("output.png")
    let io = try NativeImageIO()
    try await io.write(NativeOpticalFlowTests.texture(width: 128, height: 96, dx: 0, dy: 0), to: input)
    let preview = try NativeMediaPreview()
    var options = MediaProcessingOptions(renderingModel: URL(fileURLWithPath: package))
    options.detailStrength = 2
    options.profile = .natural
    let request = MediaPreviewRequest(input: input, isVideo: false, options: options)
    let first = try await preview.render(request)
    _ = try await NativeMediaProcessor().processImage(input: input, output: output, options: options)
    let export = try await io.displayImage(io.read(output))
    let firstBytes = bytes(first.processed), exportBytes = bytes(export)
    XCTAssertEqual(firstBytes.count, exportBytes.count)
    XCTAssertLessThanOrEqual(zip(firstBytes, exportBytes).map { abs(Int($0) - Int($1)) }.max()!, 1)
    options.intensity = 0
    options.profile = .standard
    let bypass = try await preview.render(.init(input: input, isVideo: false, options: options))
    XCTAssertEqual(bytes(bypass.processed), bytes(bypass.original))
    XCTAssertNotEqual(firstBytes, bytes(bypass.processed))
    let repeated = try await preview.render(request)
    XCTAssertEqual(bytes(repeated.processed), firstBytes)
    XCTAssertEqual(repeated.historyFrames, 0)
    var invalidModel = request.options
    invalidModel.renderingModel = directory.appendingPathComponent("missing.dlssmodel")
    do {
      _ = try await preview.render(.init(input: input, isVideo: false, options: invalidModel))
      XCTFail("An invalid model selection must report an error")
    } catch {}
    let recovered = try await preview.render(request)
    XCTAssertEqual(bytes(recovered.processed), firstBytes)
  }

  private func bytes(_ image: CGImage) -> [UInt8] {
    let data = image.dataProvider!.data! as Data
    return Array(data)
  }

  private func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("mlxdlss-preview-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
  }
}
