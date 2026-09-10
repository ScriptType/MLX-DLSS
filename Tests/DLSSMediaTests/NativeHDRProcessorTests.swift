import CoreMedia
import CoreVideo
import DLSSMLX
import Foundation
import XCTest
@testable import DLSSMedia

final class NativeHDRProcessorTests: XCTestCase, @unchecked Sendable {
  func testBypassRetainsOriginalAndSameFrameDiagnosticViews() async throws {
    let processor = try NativeHDRProcessor(configuration: .init(strength: 0))
    let original = try frame(index: 0)
    let result = try await processor.process(original)
    XCTAssertTrue(result.enhanced === original.original)
    XCTAssertTrue(result.identity === original.original)
    XCTAssertFalse(result.usedModel)
    XCTAssertEqual(result.metadata.time, original.metadata.time)
    XCTAssertEqual(result.metadata.generation, original.metadata.generation)
    XCTAssertGreaterThan(values(result.enhanced).max()!, 203)
    XCTAssertTrue(values(result.proxy).allSatisfy { (0...1).contains($0) })
  }

  func testRealTinyModelSequencePreservesHDRAndResetsGeneration() async throws {
    guard let path = ProcessInfo.processInfo.environment["MLXDLSS_NEURAL_RENDERING_PACKAGE"] else {
      throw XCTSkip("Set MLXDLSS_NEURAL_RENDERING_PACKAGE to validate the real model")
    }
    let processor = try NativeHDRProcessor(configuration: .init(modelURL: URL(fileURLWithPath: path),
      processingWidth: 64, processingHeight: 40, strength: 0.6, colorStrength: 0.5, motion: .zero))
    for index in 0..<3 {
      let source = try frame(index: UInt64(index), generation: index == 2 ? 2 : 1)
      let result = try await processor.process(source)
      XCTAssertTrue(result.usedModel)
      XCTAssertEqual(result.historyReset, index != 1)
      XCTAssertEqual(result.metadata.time, source.metadata.time)
      XCTAssertEqual(result.enhanced.width, 96)
      XCTAssertEqual(result.enhanced.height, 64)
      XCTAssertTrue(values(result.enhanced).allSatisfy(\.isFinite))
      XCTAssertGreaterThan(values(result.enhanced).max()!, 203)
      let error = zip(values(result.identity), values(source.original)).map { abs($0-$1) }.max()!
      XCTAssertLessThan(error, 0.001)
      XCTAssertGreaterThan(result.timings.inferenceSeconds, 0)
      print("native-hdr-model frame=\(index) inferenceCompletedWall=\(result.timings.inferenceSeconds) maximumNits=\(values(result.enhanced).max()!) identityMaxError=\(error)")
    }
    try await processor.configure(strength: 0, colorStrength: 0.5, processingWidth: 48, processingHeight: 32)
    let source = try frame(index: 3, generation: 2)
    let bypass = try await processor.process(source)
    XCTAssertTrue(bypass.enhanced === source.original)
    XCTAssertFalse(bypass.usedModel)
  }

  func testNativeReaderImportsSDRHDR10AndHLGFixtures() async throws {
    guard #available(macOS 26.0, *) else { throw XCTSkip("Native video requires macOS 26") }
    guard let directory = ProcessInfo.processInfo.environment["MLXDLSS_HDR_FIXTURES"] else {
      throw XCTSkip("Set MLXDLSS_HDR_FIXTURES to generated SDR/PQ/HLG fixture directory")
    }
    for (file, transfer) in [("sdr-24.mp4", MLXHDRColorMetadata.Transfer.bt1886), ("hdr10-30.mp4", .pq), ("hlg-60.mp4", .hlg)] {
      let reader = try await NativeHDRVideoReader(url: URL(fileURLWithPath: directory).appendingPathComponent(file),
        sourceID: "fixture-\(file)", generation: 11)
      let firstFrame = try await reader.next()
      let first = try XCTUnwrap(firstFrame)
      let secondFrame = try await reader.next()
      let second = try XCTUnwrap(secondFrame)
      XCTAssertEqual(first.metadata.color.transfer, transfer)
      XCTAssertEqual(first.metadata.generation, 11)
      XCTAssertEqual(first.original.width, 320)
      XCTAssertEqual(first.original.height, 192)
      XCTAssertGreaterThan(second.metadata.time, first.metadata.time)
      XCTAssertEqual(second.metadata.time - first.metadata.time, first.metadata.duration)
      XCTAssertEqual(second.metadata.frameIndex, 1)
      let decoded = try await reader.nextDecoded()
      XCTAssertEqual(decoded?.metadata.frameIndex, 2)
      XCTAssertEqual(decoded?.metadata.color.transfer, transfer)
      XCTAssertEqual(decoded.map { CVPixelBufferGetPlaneCount($0.pixelBuffer.buffer) }, 2)
      XCTAssertNotNil(first.sourcePixelBuffer)
      XCTAssertTrue(values(first.original).allSatisfy(\.isFinite))
      XCTAssertGreaterThan(values(first.original).max()!, 1)
      if file == "hdr10-30.mp4" { XCTAssertNotNil(first.metadata.color.masteringDisplay) }
      print("native-hdr-decode file=\(file) maxNits=\(values(first.original).max()!) importGPUSeconds=\(first.importGPUSeconds.map(String.init(describing:)) ?? "unavailable") assumptions=\(first.metadata.color.assumptions)")
      await reader.cancel()
    }
  }

  private func frame(index: UInt64, generation: UInt64 = 1) throws -> MLXHDRFrame {
    let width = 96, height = 64
    let pixels: [Float] = (0..<width*height*3).map {
      let x = ($0/3)%width, c = $0%3
      if x < width/3 { return c == 0 ? 1000 : 2 }
      if x < 2*width/3 { return Float(x-32)*30 + 0.01 }
      return c == 1 ? 800 : 20
    }
    let image = try pixels.withUnsafeBytes { try MLXVideoFrame(rgb: Data($0), width: width, height: height) }
    return MLXHDRFrame(original: image, metadata: .init(time: CMTime(value: Int64(index)*1001, timescale: 30000),
      duration: CMTime(value: 1001, timescale: 30000), sourceID: "synthetic", frameIndex: index,
      generation: generation, color: .init(transfer: .pq, primaries: .bt2020, matrix: .bt2020)))
  }
  private func values(_ frame: MLXVideoFrame) -> [Float] {
    frame.copyRGBData().withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  }
}
