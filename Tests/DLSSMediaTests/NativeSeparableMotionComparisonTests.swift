import CoreVideo
import CryptoKit
import Foundation
import IOSurface
import MLX
import XCTest
@testable import DLSSMedia
@testable import DLSSMLX

/// Restores existing completed VT buffers; never estimates new optical flow.
final class NativeSeparableMotionComparisonTests: XCTestCase, @unchecked Sendable {
  private let costSHA = "4bbca00854a99638a7a007a9ef54afe886dac49284c9354bb92f7ceb65bbdd5f"
  private let inputSHA = "3c376d69b898a7fa0ee2474e233438c9071dbe3d63b3cece57d9b8e1bfa68181"
  private let sourceSHA = "d8878517b8e68ce8971c3c8e1813c92796303937ebde387cd875545b7b4e313e"
  private let width = 1920, height = 1080

  private struct Time: Codable, Equatable { let value: Int64; let timescale: Int32 }
  private struct Pin: Decodable { let path: String; let bytes: Int; let sha256: String }
  private struct Hash: Decodable { let bytes: Int; let sha256: String }
  private struct Source: Decodable {
    let sourceFrameIndex: Int
    let pts: Time
    let duration: Time
    let path: String
    let bytes: Int
    let sha256: String
  }
  private struct Inputs: Decodable {
    let width: Int
    let height: Int
    let frames: [Source]
    let captureManifest: Pin
    let source: Pin
  }
  private struct Flow: Decodable {
    let width: Int
    let height: Int
    let pixelFormat: UInt32
    let originalBytesPerRow: Int
    let logicalBytesPerRow: Int
    let units: String
    let payload: Pin
  }
  private struct Expected: Decodable {
    let vectors: Hash
    let confidence: Hash
    let reliableFractionBits: UInt32
    let warpedLumaErrorBits: UInt32
    let reset: Bool
  }
  private struct Captured: Decodable {
    let sourceFrameIndex: Int
    let previousSourceFrameIndex: Int?
    let pts: Time
    let duration: Time
    let inputSHA256: String
    let hasMotion: Bool
    let productionMotion: Expected?
    let forwardFlow: Flow?
    let backwardFlow: Flow?
    let randomAccessSubmitted: Bool?
    let sameVTBuffersFullMotionExact: Bool?
  }
  private struct Cost: Decodable {
    let passed: Bool
    let completedPhase: String
    let frozenFilesUnchanged: Bool
    let width: Int
    let height: Int
    let captureFrames: [Captured]
    let frozenSHA256Before: [String: String]
  }
  private struct Restored {
    let owner: MLXPixelBuffer
    let bytes: Data
    let original: Flow
    let url: URL
  }
  private struct Pair {
    let source: Source
    let previous: Source
    let current: MLXVideoFrame
    let prior: MLXVideoFrame
    let forward: Restored
    let backward: Restored
    let expected: Expected
    let historicalRandomAccess: Bool
  }
  /// CPU-owned copies only. No motion object or its MLX output survives a call.
  private struct Result {
    let vectors: Data
    let confidence: Data
    let reliableBits: UInt32
    let errorBits: UInt32
    let reset: Bool
    let started: UInt64
    let ended: UInt64
    let memory: [String: UInt64]
    var seconds: Double { Double(ended - started) / 1e9 }
    func matches(_ other: Self) -> Bool {
      vectors == other.vectors && confidence == other.confidence &&
        reliableBits == other.reliableBits && errorBits == other.errorBits && reset == other.reset
    }
  }
  private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "NativeSeparableMotionComparison", code: 1,
      userInfo: [NSLocalizedDescriptionKey: message]) }
  }
  private func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  private func fileDigest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while let data = try file.read(upToCount: 1024 * 1024), !data.isEmpty { hash.update(data: data) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
  private func nonfinite(_ data: Data) -> Int {
    data.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: 4).reduce(0) { count, offset in
        let bits = bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        return count + (Float(bitPattern: bits).isFinite ? 0 : 1)
      }
    }
  }
  private func time(_ value: Time) -> [String: Any] {
    ["value": value.value, "timescale": Int64(value.timescale)]
  }
  private func pin(_ url: URL, bytes: Int, sha: String) -> [String: Any] {
    ["path": url.path, "bytes": bytes, "sha256": sha]
  }
  private func save(_ data: Data, name: String, output: URL) throws -> [String: Any] {
    try data.write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
    return ["path": name, "bytes": data.count, "sha256": digest(data)]
  }
  private func snapshot(vectors: MLXArray, confidence: MLXArray, reliable: Float, error: Float,
    reset: Bool, started: UInt64, ended: UInt64) -> Result {
    let memory = MLXRuntimeDiagnostics.memorySnapshot()
    return Result(vectors: vectors.asData(access: .copy).data, confidence: confidence.asData(access: .copy).data,
      reliableBits: reliable.bitPattern, errorBits: error.bitPattern, reset: reset, started: started, ended: ended,
      memory: ["active": memory.activeBytes, "cache": memory.cacheBytes, "peakActive": memory.peakActiveBytes])
  }
  private func run(_ arm: String, pair: Pair) throws -> Result {
    // Keep branch selection and post-return payload reads outside each interval.
    if arm == "baseline" {
      let started = DispatchTime.now().uptimeNanoseconds
      let motion = try MLXVideoMotion(current: pair.current, previous: pair.prior,
        backward: pair.backward.owner, forward: pair.forward.owner, units: .flowPixels, sceneCutThreshold: 0.3)
      let ended = DispatchTime.now().uptimeNanoseconds
      return snapshot(vectors: motion.vectors, confidence: motion.confidence, reliable: motion.reliableFraction,
        error: motion.warpedLumaError, reset: motion.reset, started: started, ended: ended)
    }
    let started = DispatchTime.now().uptimeNanoseconds
    let motion = try SeparableErosionVideoMotion(current: pair.current, previous: pair.prior,
      backward: pair.backward.owner, forward: pair.forward.owner, units: .flowPixels, sceneCutThreshold: 0.3)
    let ended = DispatchTime.now().uptimeNanoseconds
    return snapshot(vectors: motion.vectors, confidence: motion.confidence, reliable: motion.reliableFraction,
      error: motion.warpedLumaError, reset: motion.reset, started: started, ended: ended)
  }
  private func matchesHistory(_ result: Result, _ expected: Expected) -> Bool {
    result.vectors.count == expected.vectors.bytes && result.confidence.count == expected.confidence.bytes &&
      digest(result.vectors) == expected.vectors.sha256 && digest(result.confidence) == expected.confidence.sha256 &&
      result.reliableBits == expected.reliableFractionBits && result.errorBits == expected.warpedLumaErrorBits &&
      result.reset == expected.reset
  }
  private func record(_ result: Result, expected: Expected, timed: Bool) -> [String: Any] {
    var row: [String: Any] = [
      "vectors": ["bytes": result.vectors.count, "sha256": digest(result.vectors), "nonfiniteScalars": nonfinite(result.vectors)],
      "confidence": ["bytes": result.confidence.count, "sha256": digest(result.confidence), "nonfiniteScalars": nonfinite(result.confidence)],
      "reliableFractionBits": result.reliableBits, "warpedLumaErrorBits": result.errorBits, "reset": result.reset,
      "scalarValuesFinite": Float(bitPattern: result.reliableBits).isFinite && Float(bitPattern: result.errorBits).isFinite,
      "historicalProductionExact": matchesHistory(result, expected), "mlxBytes": result.memory]
    if timed {
      row["startUptimeNanoseconds"] = result.started; row["endUptimeNanoseconds"] = result.ended
      row["motionCompletedWallSeconds"] = result.seconds
    }
    return row
  }
  private func mismatchPayloads(_ result: Result, prefix: String, output: URL) throws -> [String: Any] {
    ["vectors": try save(result.vectors, name: "\(prefix)-vectors.f32", output: output),
      "confidence": try save(result.confidence, name: "\(prefix)-confidence.f32", output: output)]
  }
  private func logicalFlow(_ owner: MLXPixelBuffer) throws -> Data {
    let buffer = owner.buffer, row = owner.width * 4, stride = CVPixelBufferGetBytesPerRow(owner.buffer)
    try require(!CVPixelBufferIsPlanar(buffer) && stride >= row && stride.isMultiple(of: 2), "Invalid restored flow stride")
    try require(CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess, "Cannot read restored flow")
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let address = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer))
    var bytes = Data(count: row * owner.height)
    bytes.withUnsafeMutableBytes { destination in
      for y in 0..<owner.height {
        destination.baseAddress!.advanced(by: y * row).copyMemory(
          from: address.advanced(by: y * stride), byteCount: row)
      }
    }
    return bytes
  }
  private func restore(_ flow: Flow, base: URL) throws -> Restored {
    try require(flow.width == 240 && flow.height == 135 && flow.pixelFormat == kCVPixelFormatType_TwoComponent16Half &&
      flow.logicalBytesPerRow == 960 && flow.originalBytesPerRow >= 960 && flow.originalBytesPerRow.isMultiple(of: 2) &&
      flow.units == "flowPixels" && flow.payload.bytes == 129600, "Retained flow format/geometry differs")
    let url = base.appendingPathComponent(flow.payload.path).resolvingSymlinksInPath()
    try require(!flow.payload.path.hasPrefix("/") && url.path.hasPrefix(base.path + "/"), "Flow must be contained in cost output")
    let bytes = try Data(contentsOf: url)
    try require(bytes.count == flow.payload.bytes && digest(bytes) == flow.payload.sha256, "Retained flow payload differs")
    let owner = try NativePixelBuffers.make(width: flow.width, height: flow.height, format: flow.pixelFormat)
    let buffer = owner.buffer, stride = CVPixelBufferGetBytesPerRow(buffer)
    try require(owner.width == flow.width && owner.height == flow.height &&
      CVPixelBufferGetPixelFormatType(buffer) == flow.pixelFormat && !CVPixelBufferIsPlanar(buffer) &&
      CVPixelBufferGetIOSurface(buffer) != nil && stride >= flow.logicalBytesPerRow &&
      stride.isMultiple(of: 2), "Restored flow lacks required CV/IOSurface layout")
    // Complete and unlock CPU writes before creating either lazy MLX consumer.
    try require(CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess, "Cannot write restored flow")
    do {
      defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
      let address = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer))
      bytes.withUnsafeBytes { source in
        for y in 0..<flow.height {
          address.advanced(by: y * stride).copyMemory(
            from: source.baseAddress!.advanced(by: y * flow.logicalBytesPerRow), byteCount: flow.logicalBytesPerRow)
        }
      }
    }
    try require(logicalFlow(owner) == bytes, "Restored logical flow bytes differ")
    return Restored(owner: owner, bytes: bytes, original: flow, url: url)
  }
  private func flowRecord(_ value: Restored) -> [String: Any] {
    ["payload": pin(value.url, bytes: value.bytes.count, sha: value.original.payload.sha256),
      "width": value.owner.width, "height": value.owner.height,
      "pixelFormat": CVPixelBufferGetPixelFormatType(value.owner.buffer),
      "originalBytesPerRow": value.original.originalBytesPerRow,
      "restoredBytesPerRow": CVPixelBufferGetBytesPerRow(value.owner.buffer),
      "logicalBytesPerRow": value.original.logicalBytesPerRow,
      "importerHalfScalarStride": CVPixelBufferGetBytesPerRow(value.owner.buffer) / 2,
      "hasIOSurface": CVPixelBufferGetIOSurface(value.owner.buffer) != nil,
      "units": "flowPixels", "writesCompletedAndUnlocked": true,
      "restoredLogicalSHA256Before": digest(value.bytes), "restoredLogicalByteEqualBefore": true]
  }
  private func verifyOwners(_ pair: Pair) throws {
    try require(logicalFlow(pair.forward.owner) == pair.forward.bytes && logicalFlow(pair.backward.owner) == pair.backward.bytes,
      "Restored logical flow mutated")
    try require(digest(pair.current.copyRGBData()) == pair.source.sha256 &&
      digest(pair.prior.copyRGBData()) == pair.previous.sha256, "Materialized source frame mutated")
  }
  private func distribution(_ values: [Double]) -> [String: Any] {
    guard !values.isEmpty else { return ["count": 0] }
    let sorted = values.sorted(), middle = values.count / 2
    let median = values.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    return ["count": values.count, "meanSeconds": values.reduce(0, +) / Double(values.count),
      "medianSeconds": median, "p95Seconds": sorted[Int(ceil(Double(values.count) * 0.95)) - 1],
      "minimumSeconds": sorted[0], "maximumSeconds": sorted[sorted.count - 1]]
  }

  func testMatchedFullMotion() throws {
    let env = ProcessInfo.processInfo.environment
    guard let outputPath = env["MLXDLSS_SEPARABLE_MOTION_OUTPUT"] else {
      throw XCTSkip("Opt-in retained-flow full-motion comparison")
    }
    #if DEBUG
    throw NSError(domain: "NativeSeparableMotionComparison", code: 2,
      userInfo: [NSLocalizedDescriptionKey: "Comparison requires release configuration"])
    #endif
    let costURL = URL(fileURLWithPath: try XCTUnwrap(env["MLXDLSS_SEPARABLE_MOTION_INPUT_REPORT"])).standardizedFileURL
    let inputURL = URL(fileURLWithPath: try XCTUnwrap(env["MLXDLSS_SEPARABLE_MOTION_INPUTS"])).standardizedFileURL
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { root.deleteLastPathComponent() }
    let output = URL(fileURLWithPath: outputPath).standardizedFileURL
    try require(!FileManager.default.fileExists(atPath: output.path) &&
      (try? output.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true, "Require new output directory")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    var report: [String: Any] = ["schemaVersion": 1, "passed": false,
      "actualBackend": "retained-flow-full-motion-metal-replay", "modelLoaded": false, "newOpticalFlowEstimated": false,
      "completedPhase": "initializing", "expectedProgressUnits": 99, "width": width, "height": height,
      "flowWidth": 240, "flowHeight": 135, "flowUnits": "flowPixels", "sceneCutThresholdBits": Float(0.3).bitPattern,
      "restoreAttributes": ["IOSurfaceProperties": "empty dictionary", "MetalCompatibility": true, "format": "TwoComponent16Half"],
      "timingScope": "Constructor through existing joint blocking eval and scalar reads; restore/materialize before interval, output Data/hash/publication afterward",
      "limitations": ["Preflight already executes both arms. Four measured passes are not cold/compilation timing.",
        "MLXVideoMotion is stateless; captured reset identity does not replay NativeOpticalFlow session transitions.",
        "Historical VT buffers retain earlier half/CI/filter decisions; no new VT estimation or model inference.",
        "Different modules/kernel names and outside-timer readback/hash/report work limit kernel-only attribution.",
        "Full NativeOpticalFlow.prepare benefit, temporal quality, Live, M5 and hard memory limits remain unestablished.",
        "Results retain CPU snapshots for one pair only; source frames and22 immutable restored flow owners remain until verification."]]
    var frozen = [String: String](), sourceRows = [[String: Any]](), restoredRows = [[String: Any]]()
    var preflight = [[String: Any]](), samples = [[String: Any]](), comparisons = [[String: Any]]()
    var completedPreflightInitializers = 0
    func publish() throws {
      var value = report
      value["frozenSHA256Before"] = frozen
      value["sourceFrames"] = sourceRows; value["restoredPairs"] = restoredRows
      value["preflight"] = preflight; value["samples"] = samples; value["comparisons"] = comparisons
      value["progressUnits"] = preflight.count + samples.count
      value["preflightInitializers"] = completedPreflightInitializers
      value["timedInitializers"] = samples.count; value["measuredCalls"] = samples.count
      value["measuredPairs"] = comparisons.count; value["measuredNaturalPairs"] = comparisons.count
      try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        .write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    let priorCache = MLXRuntimeDiagnostics.cacheLimitBytes
    report["priorMLXCachePolicyBytes"] = priorCache
    defer {
      do {
        try MLXRuntimeDiagnostics.setCacheLimitBytes(priorCache)
        report["restoredMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
        if MLXRuntimeDiagnostics.cacheLimitBytes != priorCache { report["passed"] = false; XCTFail("Cache policy differs after restoration") }
      } catch {
        report["passed"] = false; report["cacheRestorationError"] = error.localizedDescription
        XCTFail("Cache restoration failed: \(error)")
      }
      var after = [String: String]()
      for (path, expected) in frozen {
        do { after[path] = try fileDigest(URL(fileURLWithPath: path)) }
        catch { report["hashVerificationError"] = error.localizedDescription }
        if after[path] != expected { report["passed"] = false; XCTFail("Frozen file changed: \(path)") }
      }
      report["frozenSHA256After"] = after
      report["frozenFilesUnchanged"] = !frozen.isEmpty && after == frozen
      do { try publish() } catch { XCTFail("Final report publication failed: \(error)") }
    }
    do {
      let costBytes = try Data(contentsOf: costURL), inputBytes = try Data(contentsOf: inputURL)
      try require(digest(costBytes) == costSHA && digest(inputBytes) == inputSHA, "Pinned cost/input reports differ")
      frozen[costURL.path] = costSHA; frozen[inputURL.path] = inputSHA
      report["inputReport"] = pin(costURL, bytes: costBytes.count, sha: costSHA)
      report["inputManifest"] = pin(inputURL, bytes: inputBytes.count, sha: inputSHA)
      let cost = try JSONDecoder().decode(Cost.self, from: costBytes)
      let inputs = try JSONDecoder().decode(Inputs.self, from: inputBytes)
      try require(cost.passed && cost.completedPhase == "complete" && cost.frozenFilesUnchanged &&
        cost.width == width && cost.height == height && inputs.width == width && inputs.height == height &&
        inputs.frames.map(\.sourceFrameIndex) == Array(1496...1507) &&
        cost.captureFrames.map(\.sourceFrameIndex) == Array(1496...1507), "Retained source/capture inventory differs")
      for (path, expected) in cost.frozenSHA256Before where
        path.contains("/Sources/") || path.contains("/Tests/DLSSMediaTests/") ||
        path.hasPrefix(root.appendingPathComponent(".build/debug/").path + "/") {
        try require(fileDigest(URL(fileURLWithPath: path)) == expected, "Measured production source/runtime differs: \(path)")
        frozen[path] = expected
      }
      let productionURL = root.appendingPathComponent("vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoMotion.swift")
      let candidateURL = root.appendingPathComponent("vendor/MLX-DLSS/Tests/DLSSMediaTests/SeparableErosionVideoMotion.swift")
      let productionBytes = try Data(contentsOf: productionURL)
      var candidate = try String(contentsOf: candidateURL, encoding: .utf8)
      let importBlock = "// SEPARABLE-MOTION-IMPORT-BEGIN\n@testable import DLSSMLX\n// SEPARABLE-MOTION-IMPORT-END\n"
      let callBlock = "    // SEPARABLE-MOTION-EROSION-BEGIN\n    confidence = SeparableMotionErosion.apply(quality[0], params: params)\n    // SEPARABLE-MOTION-EROSION-END\n"
      let originalCall = "    confidence = Self.erode([quality[0], params], grid: (count, 1, 1), threadGroup: (256, 1, 1),\n      outputShapes: [quality[0].shape], outputDTypes: [.float32])[0]\n"
      try require(candidate.components(separatedBy: importBlock).count == 2 &&
        candidate.components(separatedBy: callBlock).count == 2, "Candidate markers differ")
      candidate = candidate.replacingOccurrences(of: importBlock, with: "").replacingOccurrences(of: callBlock, with: originalCall)
      for (copy, original) in [
        ("SeparableErosionVideoMotion", "MLXVideoMotion"),
        ("mlxdlss_separable_motion_flow_import", "mlxdlss_native_flow_import"),
        ("mlxdlss_separable_motion_quality", "mlxdlss_native_motion_quality"),
        ("mlxdlss_separable_motion_unused_erode", "mlxdlss_native_motion_erode"),
        ("mlxdlss_separable_motion_guide_resize", "mlxdlss_native_guide_resize")] {
        candidate = candidate.replacingOccurrences(of: copy, with: original)
      }
      try require(digest(productionBytes) == sourceSHA && Data(candidate.utf8) == productionBytes, "Candidate source reversal differs")
      report["reversedProductionMotionSHA256"] = digest(Data(candidate.utf8))
      let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL)
      let bundleMetallib = executable.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
      report["actualTestExecutable"] = executable.path; report["actualTestBundleMetallib"] = bundleMetallib.path
      for url in [candidateURL, URL(fileURLWithPath: #filePath), executable, bundleMetallib,
        root.appendingPathComponent("vendor/MLX-DLSS/.build/release/mlx.metallib"),
        root.appendingPathComponent("vendor/MLX-DLSS/Tests/DLSSMediaTests/SeparableMotionErosion.swift"),
        root.appendingPathComponent("vendor/MLX-DLSS/Tests/DLSSMediaTests/NativeMotionErosionComparisonTests.swift")] {
        frozen[url.path] = try fileDigest(url)
      }
      for entry in [inputs.captureManifest, inputs.source] {
        let url = entry.path.hasPrefix("/") ? URL(fileURLWithPath: entry.path) : root.appendingPathComponent(entry.path)
        try require(fileDigest(url) == entry.sha256 && url.resourceValues(forKeys: [.fileSizeKey]).fileSize == entry.bytes,
          "Historical source/capture pin differs")
        frozen[url.path] = entry.sha256
      }
      try MLXRuntimeDiagnostics.setCacheLimitBytes(256 * 1024 * 1024)
      report["benchmarkMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
      report["completedPhase"] = "restore-inputs"; try publish()
      var frames = [MLXVideoFrame]()
      for (ordinal, source) in inputs.frames.enumerated() {
        let captured = cost.captureFrames[ordinal]
        try require(captured.pts == source.pts && captured.duration == source.duration && captured.inputSHA256 == source.sha256 &&
          source.pts.timescale == 24000 && source.duration.timescale == 24000 && source.duration.value == 1001,
          "Exact source timing/hash differs from capture")
        let url = source.path.hasPrefix("/") ? URL(fileURLWithPath: source.path) : root.appendingPathComponent(source.path)
        let bytes = try Data(contentsOf: url)
        try require(bytes.count == width * height * 12 && bytes.count == source.bytes && digest(bytes) == source.sha256 &&
          nonfinite(bytes) == 0, "Source payload extent/hash/finite domain differs")
        frozen[url.path] = source.sha256
        let frame = try MLXVideoFrame(rgb: bytes, width: width, height: height)
        try require(frame.copyRGBData() == bytes, "Materialized source bytes differ")
        frames.append(frame)
        sourceRows.append(["sourceFrameIndex": source.sourceFrameIndex, "pts": time(source.pts), "duration": time(source.duration),
          "payload": pin(url, bytes: bytes.count, sha: source.sha256), "materializedByteEqual": true])
        try publish()
      }
      let first = cost.captureFrames[0]
      try require(!first.hasMotion && first.productionMotion == nil && first.forwardFlow == nil && first.backwardFlow == nil,
        "Cold1496 unexpectedly owns motion")
      let base = costURL.deletingLastPathComponent().resolvingSymlinksInPath()
      var pairs = [Pair]()
      for ordinal in 1..<inputs.frames.count {
        let captured = cost.captureFrames[ordinal], source = inputs.frames[ordinal], previous = inputs.frames[ordinal - 1]
        try require(captured.hasMotion && captured.sameVTBuffersFullMotionExact == true &&
          captured.previousSourceFrameIndex == previous.sourceFrameIndex, "Captured pair ownership differs")
        let expected = try XCTUnwrap(captured.productionMotion)
        try require(expected.vectors.bytes == width * height * 8 && expected.confidence.bytes == width * height * 4 &&
          Float(bitPattern: expected.reliableFractionBits).isFinite && Float(bitPattern: expected.warpedLumaErrorBits).isFinite,
          "Historical motion extent/scalars differ")
        let forward = try restore(XCTUnwrap(captured.forwardFlow), base: base)
        let backward = try restore(XCTUnwrap(captured.backwardFlow), base: base)
        frozen[forward.url.path] = forward.original.payload.sha256
        frozen[backward.url.path] = backward.original.payload.sha256
        let pair = Pair(source: source, previous: previous, current: frames[ordinal], prior: frames[ordinal - 1],
          forward: forward, backward: backward, expected: expected,
          historicalRandomAccess: try XCTUnwrap(captured.randomAccessSubmitted))
        pairs.append(pair)
        restoredRows.append(["sourceFrameIndex": source.sourceFrameIndex, "previousSourceFrameIndex": previous.sourceFrameIndex,
          "pts": time(source.pts), "duration": time(source.duration), "historicalRandomAccessSubmitted": pair.historicalRandomAccess,
          "forward": flowRecord(forward), "backward": flowRecord(backward),
          "expectedVectorsSHA256": expected.vectors.sha256, "expectedConfidenceSHA256": expected.confidence.sha256,
          "expectedReliableFractionBits": expected.reliableFractionBits, "expectedWarpedLumaErrorBits": expected.warpedLumaErrorBits,
          "expectedReset": expected.reset])
        try publish()
      }
      report["restoredFlowCount"] = pairs.count * 2
      report["completedPhase"] = "historical-preflight"; try publish()
      for pair in pairs {
        report["currentSourceFrameIndex"] = pair.source.sourceFrameIndex; report["currentArm"] = "baseline"
        try publish()
        let baseline = try run("baseline", pair: pair)
        completedPreflightInitializers += 1
        var row: [String: Any] = ["sourceFrameIndex": pair.source.sourceFrameIndex,
          "phase": "preflight-unmeasured", "baseline": record(baseline, expected: pair.expected, timed: false)]
        // The actual baseline must reproduce historical data before this pair's candidate is invoked.
        if !matchesHistory(baseline, pair.expected) {
          row["baselineFailurePayloads"] = try mismatchPayloads(baseline, prefix: "preflight-\(pair.source.sourceFrameIndex)-baseline", output: output)
          report["incompletePreflightPair"] = row; try publish()
          try require(false, "Production baseline fails historical reproduction; no candidate claim")
        }
        report["currentArm"] = "candidate"; try publish()
        let candidate = try run("candidate", pair: pair)
        completedPreflightInitializers += 1
        row["candidate"] = record(candidate, expected: pair.expected, timed: false)
        row["baselineCandidateFullExact"] = baseline.matches(candidate)
        if !baseline.matches(candidate) {
          row["baselineFailurePayloads"] = try mismatchPayloads(baseline, prefix: "preflight-\(pair.source.sourceFrameIndex)-baseline", output: output)
          row["candidateFailurePayloads"] = try mismatchPayloads(candidate, prefix: "preflight-\(pair.source.sourceFrameIndex)-candidate", output: output)
        }
        preflight.append(row); try publish()
        try require(baseline.matches(candidate) && matchesHistory(candidate, pair.expected), "Preflight full-motion mismatch; payloads retained")
        try verifyOwners(pair)
      }
      report["preflightFinishedBeforeTiming"] = true
      report["completedPhase"] = "measured-full-motion"; try publish()
      for pass in 0..<4 {
        let order = pass.isMultiple(of: 2) ? Array(pairs.indices) : Array(pairs.indices.reversed())
        for (orderIndex, index) in order.enumerated() {
          let pair = pairs[index], pairIndex = comparisons.count
          let arms = (pass + index).isMultiple(of: 2) ? ["baseline", "candidate"] : ["candidate", "baseline"]
          var results = [String: Result]()
          report["currentSourceFrameIndex"] = pair.source.sourceFrameIndex; report["currentPass"] = pass
          for (armIndex, arm) in arms.enumerated() {
            report["currentArm"] = arm; try publish()
            let result = try run(arm, pair: pair)
            results[arm] = result
            var row = record(result, expected: pair.expected, timed: true)
            row["pairIndex"] = pairIndex; row["pass"] = pass; row["phase"] = "measured"; row["measured"] = true
            row["maskProvenance"] = "natural"; row["arm"] = arm; row["armIndex"] = armIndex
            row["sourceFrameIndex"] = pair.source.sourceFrameIndex; row["previousSourceFrameIndex"] = pair.previous.sourceFrameIndex
            row["sourcePairIndex"] = index; row["orderIndex"] = orderIndex
            if !matchesHistory(result, pair.expected) {
              row["failurePayloads"] = try mismatchPayloads(result, prefix: "measured-\(pass)-\(pair.source.sourceFrameIndex)-\(arm)", output: output)
            }
            samples.append(row); try publish()
          }
          let a = try XCTUnwrap(results["baseline"]), b = try XCTUnwrap(results["candidate"])
          var row: [String: Any] = ["pairIndex": pairIndex, "pass": pass, "phase": "measured", "measured": true,
            "maskProvenance": "natural", "sourceFrameIndex": pair.source.sourceFrameIndex, "sourcePairIndex": index,
            "orderIndex": orderIndex, "firstArm": arms[0], "baselineCandidateFullExact": a.matches(b),
            "baselineHistoricalExact": matchesHistory(a, pair.expected), "candidateHistoricalExact": matchesHistory(b, pair.expected),
            "baselineSeconds": a.seconds, "candidateSeconds": b.seconds, "candidateMinusBaselineSeconds": b.seconds - a.seconds,
            "baselineReset": a.reset, "candidateReset": b.reset]
          // Preserve both sides for independent numerical replay, including the
          // historical-matching arm whose vectors otherwise have only a saved hash.
          if !a.matches(b) {
            row["baselineFailurePayloads"] = try mismatchPayloads(a, prefix: "pair-\(pass)-\(pair.source.sourceFrameIndex)-baseline", output: output)
            row["candidateFailurePayloads"] = try mismatchPayloads(b, prefix: "pair-\(pass)-\(pair.source.sourceFrameIndex)-candidate", output: output)
          }
          comparisons.append(row); try publish()
          try require(a.matches(b) && matchesHistory(a, pair.expected) && matchesHistory(b, pair.expected),
            "Measured full-motion mismatch; raw records/payloads retained")
          try verifyOwners(pair)
        }
      }
      var balance = [[String: Any]](), rounds = [[String: Any]]()
      for pair in pairs {
        let rows = comparisons.filter { $0["sourceFrameIndex"] as? Int == pair.source.sourceFrameIndex }
        let a = rows.filter { $0["firstArm"] as? String == "baseline" }.count
        let b = rows.filter { $0["firstArm"] as? String == "candidate" }.count
        try require(a == 2 && b == 2, "Per-source-pair order balance differs")
        balance.append(["sourceFrameIndex": pair.source.sourceFrameIndex, "baselineFirst": a, "candidateFirst": b])
        try verifyOwners(pair)
      }
      for pass in 0..<4 {
        let rows = comparisons.filter { $0["pass"] as? Int == pass }
        rounds.append(["pass": pass, "maskProvenance": "natural",
          "baseline": distribution(rows.compactMap { $0["baselineSeconds"] as? Double }),
          "candidate": distribution(rows.compactMap { $0["candidateSeconds"] as? Double }),
          "pairedDifference": distribution(rows.compactMap { $0["candidateMinusBaselineSeconds"] as? Double })])
      }
      report["measuredOrderBalance"] = balance; report["measuredRounds"] = rounds
      var flowsAfter = [[String: Any]](), framesAfter = [[String: Any]]()
      for pair in pairs {
        for (direction, flow) in [("forward", pair.forward), ("backward", pair.backward)] {
          let bytes = try logicalFlow(flow.owner)
          flowsAfter.append(["sourceFrameIndex": pair.source.sourceFrameIndex, "direction": direction,
            "bytes": bytes.count, "logicalSHA256After": digest(bytes), "expectedSHA256": flow.original.payload.sha256,
            "logicalByteEqualAfter": bytes == flow.bytes])
          try require(bytes == flow.bytes, "Final flow logical bytes differ")
        }
      }
      for (source, frame) in zip(inputs.frames, frames) {
        let observed = digest(frame.copyRGBData())
        framesAfter.append(["sourceFrameIndex": source.sourceFrameIndex, "sha256After": observed, "expectedSHA256": source.sha256])
        try require(observed == source.sha256, "Final materialized source hash differs")
      }
      report["restoredFlowsAfter"] = flowsAfter; report["materializedFramesAfter"] = framesAfter
      report["measuredDistributions"] = [
        "maskProvenance": "natural", "baseline": distribution(comparisons.compactMap { $0["baselineSeconds"] as? Double }),
        "candidate": distribution(comparisons.compactMap { $0["candidateSeconds"] as? Double }),
        "pairedDifference": distribution(comparisons.compactMap { $0["candidateMinusBaselineSeconds"] as? Double })]
      report["historicalResetSourceIndices"] = pairs.filter(\.expected.reset).map(\.source.sourceFrameIndex)
      report["allHistoricalProductionResultsExact"] = true; report["allBaselineCandidateOutputsExact"] = true
      report["allSourceFramesAndFlowOwnersUnchanged"] = true
      try require(sourceRows.count == 12 && pairs.count == 11 && preflight.count == 11 && completedPreflightInitializers == 22 &&
        samples.count == 88 && comparisons.count == 44, "Incomplete full-motion comparison")
      report["completedPhase"] = "complete"; report["passed"] = true
      for key in ["currentSourceFrameIndex", "currentPass", "currentArm"] { report.removeValue(forKey: key) }
      try publish()
    } catch {
      report["failure"] = error.localizedDescription
      try? publish()
      throw error
    }
  }
}
