import AVFoundation
import DLSSMLX
import Foundation
import MLX
import XCTest
@testable import DLSSMedia

final class NativeMediaIOTests: XCTestCase, @unchecked Sendable {
  func testNativeImageRoundTripAndExistingOutputProtection() async throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let image = try NativeOpticalFlowTests.texture(width: 37, height: 29, dx: 0, dy: 0)
    let io = try NativeImageIO()
    let url = directory.appendingPathComponent("roundtrip.png")
    try await io.write(image, to: url)
    let decoded = try await io.read(url)
    XCTAssertEqual(decoded.width, 37)
    XCTAssertEqual(decoded.height, 29)
    let original = floats(image), actual = floats(decoded)
    XCTAssertLessThan(zip(original, actual).map { abs($0 - $1) }.max()!, 0.001)
    let before = try Data(contentsOf: url)
    do { try await io.write(image, to: url); XCTFail("Existing output must be preserved") }
    catch { XCTAssertEqual(try Data(contentsOf: url), before) }
  }

  func testNativeVideoRoundTripPreservesVariableTimestampsAndTrim() async throws {
    guard #available(macOS 26.0, *) else { throw XCTSkip("Native video requires macOS 26") }
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("roundtrip.mp4")
    let source = try NativeOpticalFlowTests.texture(width: 96, height: 64, dx: 0, dy: 0)
    let options = MediaProcessingOptions()
    let writer = try NativeVideoWriter(url: url, width: 96, height: 64, frameRate: 30, options: options, hasAudio: false)
    let ticks: [Int64] = [0, 1, 3, 4, 6]
    for tick in ticks { try await writer.append(source, at: CMTime(value: tick, timescale: 30)) }
    try await writer.finish(at: CMTime(value: 7, timescale: 30))
    let reader = try await NativeVideoReader(url: url, options: options)
    var frames: [NativeDecodedFrame] = []
    while let frame = try await reader.next() { frames.append(frame) }
    XCTAssertEqual(frames.count, ticks.count)
    for (frame, tick) in zip(frames, ticks) {
      XCTAssertEqual(frame.time.seconds, Double(tick) / 30, accuracy: 1e-5)
      XCTAssertEqual(frame.rgb.width, 96)
      XCTAssertEqual(frame.rgb.height, 64)
    }
    let difference = zip(floats(source), floats(frames[0].rgb)).map { abs($0 - $1) }
    XCTAssertLessThan(difference.reduce(0, +) / Float(difference.count), 0.035)
    var trim = options
    trim.startFrame = 1
    trim.frameLimit = 2
    let trimmed = try await NativeVideoReader(url: url, options: trim)
    var selected: [Double] = []
    while let frame = try await trimmed.next() { selected.append(frame.time.seconds) }
    XCTAssertEqual(selected.count, 2)
    XCTAssertEqual(selected[0], 1.0 / 30, accuracy: 1e-5)
    XCTAssertEqual(selected[1], 3.0 / 30, accuracy: 1e-5)
  }

  func testNativeWriterRejectsRepeatedTimestamp() async throws {
    guard #available(macOS 26.0, *) else { throw XCTSkip("Native video requires macOS 26") }
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let source = try NativeOpticalFlowTests.texture(width: 64, height: 64, dx: 0, dy: 0)
    let writer = try NativeVideoWriter(url: directory.appendingPathComponent("invalid.mp4"),
      width: 64, height: 64, frameRate: 30, options: MediaProcessingOptions(), hasAudio: false)
    try await writer.append(source, at: .zero)
    do { try await writer.append(source, at: .zero); XCTFail("Repeated timestamp must fail") }
    catch { XCTAssertTrue(error.localizedDescription.contains("timestamps")) }
    await writer.cancel()
  }

  func testAudioTrimAndSlowMotionPreservePitchAndMuxWithVideo() async throws {
    guard #available(macOS 26.0, *) else { throw XCTSkip("Native video requires macOS 26") }
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let wave = directory.appendingPathComponent("tone.wav")
    try Self.writeTone(to: wave)
    let optionalAudio = try await NativeAudioReader.open(url: wave,
      start: CMTime(value: 1, timescale: 5), timeScale: 2)
    let audio = try XCTUnwrap(optionalAudio)
    let source = try NativeOpticalFlowTests.texture(width: 64, height: 64, dx: 0, dy: 0)
    let output = directory.appendingPathComponent("audio.mp4")
    let writer = try NativeVideoWriter(url: output, width: 64, height: 64, frameRate: 30,
      options: MediaProcessingOptions(), hasAudio: true)
    let deadline = Task {
      try await Task.sleep(for: .seconds(20))
      await writer.cancel()
      await audio.cancel()
    }
    defer { deadline.cancel() }
    let audioTask = Task {
      var samples: [Int16] = []
      while let buffer = try await audio.next(upTo: .positiveInfinity) {
        buffer.withUnsafeSampleBuffer { sample in
          if let block = CMSampleBufferGetDataBuffer(sample) {
            let size = CMBlockBufferGetDataLength(block)
            var bytes = [UInt8](repeating: 0, count: size)
            XCTAssertEqual(CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: size, destination: &bytes), 0)
            samples += bytes.withUnsafeBytes { Array($0.bindMemory(to: Int16.self)) }
          }
        }
        try await writer.appendAudio(buffer)
      }
      await writer.finishAudio()
      return samples
    }
    do {
      for tick in 0..<48 { try await writer.append(source, at: CMTime(value: Int64(tick), timescale: 30)) }
      await writer.finishVideo(at: CMTime(value: 8, timescale: 5))
    } catch {
      audioTask.cancel()
      await writer.cancel()
      await audio.cancel()
      _ = try? await audioTask.value
      throw error
    }
    let samples = try await audioTask.value
    try await writer.finish(at: CMTime(value: 8, timescale: 5))
    XCTAssertEqual(Double(samples.count / 2) / 48000, 1.6, accuracy: 0.02)
    let tail = samples.suffix(5760).map { Double($0) }
    let tailRMS = sqrt(tail.reduce(0) { $0 + $1 * $1 } / Double(tail.count))
    XCTAssertGreaterThan(tailRMS, 5000, "The original tone must reach the end; silence padding must not mask a truncated tail")
    let middle = stride(from: 24000, to: samples.count - 24000, by: 2).map { samples[$0] }
    let crossings = zip(middle, middle.dropFirst()).filter { $0 <= 0 && $1 > 0 }.count
    let frequency = Double(crossings) / (Double(middle.count) / 48000)
    XCTAssertEqual(frequency, 1000, accuracy: 15)
    let asset = AVURLAsset(url: output)
    let audioTracks = try await asset.loadTracks(withMediaType: .audio)
    let videoTracks = try await asset.loadTracks(withMediaType: .video)
    XCTAssertEqual(audioTracks.count, 1)
    XCTAssertEqual(videoTracks.count, 1)
    let duration = try await asset.load(.duration)
    XCTAssertEqual(duration.seconds, 1.6, accuracy: 0.025)
  }

  static func writeTone(to url: URL) throws {
    let rate = 48000, count = rate
    let samples = (0..<count).map { Int16((sin(Double($0) * 2 * .pi * 1000 / Double(rate)) * 16000).rounded()) }
    var data = Data("RIFF".utf8)
    func u32(_ value: UInt32) { var value = value.littleEndian; withUnsafeBytes(of: &value) { data.append(contentsOf: $0) } }
    func u16(_ value: UInt16) { var value = value.littleEndian; withUnsafeBytes(of: &value) { data.append(contentsOf: $0) } }
    u32(UInt32(36 + count * 2))
    data.append(Data("WAVEfmt ".utf8)); u32(16); u16(1); u16(1)
    u32(UInt32(rate)); u32(UInt32(rate * 2)); u16(2); u16(16)
    data.append(Data("data".utf8)); u32(UInt32(count * 2))
    samples.withUnsafeBytes { data.append(contentsOf: $0) }
    try data.write(to: url)
  }

  private func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("mlxdlss-native-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
  }

  private func floats(_ frame: MLXVideoFrame) -> [Float] {
    frame.copyRGBData().withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  }
}
