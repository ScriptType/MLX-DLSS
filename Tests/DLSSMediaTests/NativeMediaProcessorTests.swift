import AVFoundation
import DLSSMLX
import Foundation
import XCTest
@testable import DLSSMedia

final class NativeMediaProcessorTests: XCTestCase, @unchecked Sendable {
  func testBothEffectOrdersPreserveTrimmedTimestampsAudioAndCancellation() async throws {
    guard #available(macOS 26.0, *) else { throw XCTSkip("Native video requires macOS 26") }
    let environment = ProcessInfo.processInfo.environment
    guard let model = environment["MLXDLSS_NEURAL_RENDERING_PACKAGE"],
      let generation = environment["MLXDLSS_FG_WEIGHTS"] else {
      throw XCTSkip("Set the native NR package and FG weights to validate complete video jobs")
    }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("mlxdlss-video-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let input = try await makeVideo(in: directory)
    let processor = NativeMediaProcessor()
    var options = MediaProcessingOptions(renderingModel: URL(fileURLWithPath: model),
      frameGenerationWeights: URL(fileURLWithPath: generation))
    options.startFrame = 2
    options.frameLimit = 4
    options.motion = .zero
    options.detailStrength = 1.7
    options.intensity = 0.45
    for order in MediaEffectOrder.allCases {
      options.order = order
      options.slowMotion = order == .generationThenRendering
      let output = directory.appendingPathComponent("\(order.rawValue).mp4")
      let result = try await processor.processVideo(input: input, output: output, options: options)
      XCTAssertEqual(result.inputFrames, 4)
      XCTAssertEqual(result.outputFrames, 7)
      XCTAssertEqual(result.motionBackend, "zero")
      let rate = options.slowMotion ? 30.0 : 60.0
      let asset = AVURLAsset(url: output)
      let audio = try await asset.loadTracks(withMediaType: .audio)
      XCTAssertEqual(audio.count, 1)
      let duration = try await asset.load(.duration).seconds
      XCTAssertEqual(duration, 7 / rate, accuracy: 0.002)
      let reader = try await NativeVideoReader(url: output, options: MediaProcessingOptions())
      var times: [Double] = []
      while let frame = try await reader.next() { times.append(frame.time.seconds) }
      XCTAssertEqual(times.count, 7)
      for (index, time) in times.enumerated() { XCTAssertEqual(time, Double(index) / rate, accuracy: 1e-5) }
    }
    let cancelled = directory.appendingPathComponent("cancelled.mp4")
    let task = Task {
      try await processor.processVideo(input: input, output: cancelled, options: options) { progress in
        if progress.inputFrames == 2 { withUnsafeCurrentTask { $0?.cancel() } }
      }
    }
    do { _ = try await task.value; XCTFail("Cancellation must stop the job") }
    catch { XCTAssertTrue(error is CancellationError, "\(error)") }
    XCTAssertFalse(FileManager.default.fileExists(atPath: cancelled.path))
    let leftovers = try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasPrefix(".cancelled-") }
    XCTAssertTrue(leftovers.isEmpty, "Cancelled export left temporary files: \(leftovers)")
  }

  @available(macOS 26.0, *)
  private func makeVideo(in directory: URL) async throws -> URL {
    let tone = directory.appendingPathComponent("tone.wav")
    try NativeMediaIOTests.writeTone(to: tone)
    let audio = try await NativeAudioReader.open(url: tone, start: .zero, timeScale: 1)!
    let end = CMTime(value: 12, timescale: 30)
    await audio.limit(to: end)
    let output = directory.appendingPathComponent("source.mp4")
    let writer = try NativeVideoWriter(url: output, width: 96, height: 64, frameRate: 30,
      options: MediaProcessingOptions(), hasAudio: true)
    let audioTask = Task {
      while let sample = try await audio.next(upTo: .positiveInfinity) { try await writer.appendAudio(sample) }
      await writer.finishAudio()
    }
    do {
      for index in 0..<12 {
        let frame = try NativeOpticalFlowTests.texture(width: 96, height: 64, dx: index, dy: 0)
        try await writer.append(frame, at: CMTime(value: Int64(index), timescale: 30))
      }
      await writer.finishVideo(at: end)
      try await audioTask.value
      try await writer.finish(at: end)
      return output
    } catch {
      audioTask.cancel()
      await audio.cancel()
      await writer.cancel()
      _ = try? await audioTask.value
      throw error
    }
  }
}
