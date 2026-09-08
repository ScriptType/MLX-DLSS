import AVFoundation
import DLSSMLX
import XCTest
@testable import DLSSMedia

final class NativeSuperResolutionTests: XCTestCase, @unchecked Sendable {
  func testImagePreviewExportAndWeightChanges() async throws {
    guard #available(macOS 26.0, *) else { throw XCTSkip("Native preview requires macOS 26") }
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let weights = try syntheticWeights(in: directory)
    let io = try NativeImageIO()
    let input = directory.appendingPathComponent("input.png")
    let output = directory.appendingPathComponent("output.png")
    try await io.write(NativeOpticalFlowTests.texture(width: 96, height: 64, dx: 0, dy: 0), to: input)
    let options = MediaProcessingOptions(superResolutionWeights: weights)
    let preview = try NativeMediaPreview()
    let request = MediaPreviewRequest(input: input, isVideo: false, options: options)
    let first = try await preview.render(request)
    XCTAssertEqual(first.processed.width, 192)
    XCTAssertEqual(first.processed.height, 128)
    XCTAssertEqual(first.original.width, 96)
    XCTAssertEqual(first.historyFrames, 0)
    _ = try await NativeMediaProcessor().processImage(input: input, output: output, options: options)
    let exported = try await io.displayImage(io.read(output))
    let actual = Array(exported.dataProvider!.data! as Data)
    let expected = Array(first.processed.dataProvider!.data! as Data)
    XCTAssertEqual(actual.count, expected.count)
    XCTAssertLessThanOrEqual(zip(actual, expected).map { abs(Int($0) - Int($1)) }.max()!, 1)
    let bypass = try await preview.render(.init(input: input, isVideo: false, options: MediaProcessingOptions()))
    XCTAssertEqual(bypass.processed.width, 96)
    var invalid = options
    invalid.superResolutionWeights = directory.appendingPathComponent("missing.safetensors")
    do {
      _ = try await preview.render(.init(input: input, isVideo: false, options: invalid))
      XCTFail("Missing weights must fail")
    } catch {}
    let recovered = try await preview.render(request)
    XCTAssertEqual(recovered.processed.dataProvider!.data! as Data, first.processed.dataProvider!.data! as Data)
    do {
      _ = try await NativeMediaProcessor().processImage(input: input, output: output, options: options)
      XCTFail("Existing output must be preserved")
    } catch {}
  }

  func testVideoDoublesSizeAndPreservesFramesAndTimestamps() async throws {
    guard #available(macOS 26.0, *) else { throw XCTSkip("Native video requires macOS 26") }
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let input = directory.appendingPathComponent("input.mp4")
    let writer = try NativeVideoWriter(url: input, width: 96, height: 64, frameRate: 10,
      options: MediaProcessingOptions(), hasAudio: false)
    for index in 0..<3 {
      try await writer.append(NativeOpticalFlowTests.texture(width: 96, height: 64, dx: index, dy: 0),
        at: CMTime(value: Int64(index), timescale: 10))
    }
    try await writer.finish(at: CMTime(value: 3, timescale: 10))
    let options = MediaProcessingOptions(superResolutionWeights: try syntheticWeights(in: directory))
    let output = directory.appendingPathComponent("output.mp4")
    let result = try await NativeMediaProcessor().processVideo(input: input, output: output, options: options)
    XCTAssertEqual(result.inputFrames, 3)
    XCTAssertEqual(result.outputFrames, 3)
    XCTAssertEqual(result.motionBackend, "disabled")
    XCTAssertGreaterThan(result.timing?.superResolutionSeconds ?? 0, 0)
    let reader = try await NativeVideoReader(url: output, options: MediaProcessingOptions())
    var times: [Double] = []
    while let frame = try await reader.next() {
      XCTAssertEqual(frame.rgb.width, 192)
      XCTAssertEqual(frame.rgb.height, 128)
      times.append(frame.time.seconds)
    }
    XCTAssertEqual(times.count, 3)
    for (index, time) in times.enumerated() { XCTAssertEqual(time, Double(index) / 10, accuracy: 1e-5) }
    let preview = try await NativeMediaPreview().render(.init(input: input, isVideo: true, time: 0.1, options: options))
    XCTAssertEqual(preview.processed.width, 192)
    XCTAssertEqual(preview.time, 0.1, accuracy: 1e-5)
  }

  private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxdlss-vsr-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  private func syntheticWeights(in directory: URL) throws -> URL {
    var header: [String: Any] = [:]
    var offset = 0
    for (name, shape) in VideoSuperResolver.weightShapes.sorted(by: { $0.key < $1.key }) {
      let size = shape.reduce(1, *) * 2
      header[name] = ["dtype": "F16", "shape": shape, "data_offsets": [offset, offset + size]]
      offset += size
    }
    var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    json.append(Data(repeating: 0x20, count: (8 - json.count % 8) % 8))
    var size = UInt64(json.count).littleEndian
    var data = withUnsafeBytes(of: &size) { Data($0) }
    data.append(json)
    data.append(Data(repeating: 0, count: offset))
    let url = directory.appendingPathComponent("synthetic.safetensors")
    try data.write(to: url)
    return url
  }
}
