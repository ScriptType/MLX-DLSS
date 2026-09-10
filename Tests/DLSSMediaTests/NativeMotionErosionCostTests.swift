import CoreVideo
import CryptoKit
import Foundation
import MLX
import VideoToolbox
import XCTest
@testable import DLSSMedia
@testable import DLSSMLX

private func erosionRequire(_ condition: Bool, _ message: String) throws {
  if !condition { throw NSError(domain: "NativeMotionErosionCost", code: 1,
    userInfo: [NSLocalizedDescriptionKey: message]) }
}

struct ErosionProbeRawFlow: Sendable {
  let bytes: Data
  let width: Int
  let height: Int
  let originalRowBytes: Int
  let pixelFormat: OSType

  init(_ owner: MLXPixelBuffer) throws {
    let buffer = owner.buffer
    width = owner.width; height = owner.height
    pixelFormat = CVPixelBufferGetPixelFormatType(buffer)
    originalRowBytes = CVPixelBufferGetBytesPerRow(buffer)
    try erosionRequire(!CVPixelBufferIsPlanar(buffer) && width == 240 && height == 135 &&
      pixelFormat == kCVPixelFormatType_TwoComponent16Half && originalRowBytes >= width * 4,
      "Expected actual240x135 RG16F flow")
    let status = CVPixelBufferLockBaseAddress(buffer, .readOnly)
    try erosionRequire(status == kCVReturnSuccess, "Cannot lock completed VT flow")
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let address = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer))
    var copy = Data(count: width * height * 4)
    let row = width * 4, stride = originalRowBytes, rows = height
    copy.withUnsafeMutableBytes { destination in
      for y in 0..<rows {
        destination.baseAddress!.advanced(by: y * row).copyMemory(
          from: address.advanced(by: y * stride), byteCount: row)
      }
    }
    bytes = copy
  }
}

fileprivate struct ErosionProbeMotion: Sendable {
  let vectors: Data
  let confidence: Data
  let reliableFraction: Float
  let warpedLumaError: Float
  let reset: Bool

  var finite: Bool {
    [vectors, confidence].allSatisfy { bytes in
      bytes.withUnsafeBytes { $0.bindMemory(to: Float.self).allSatisfy(\.isFinite) }
    } && reliableFraction.isFinite && warpedLumaError.isFinite
  }
  func matches(_ other: Self) -> Bool {
    vectors == other.vectors && confidence == other.confidence && reset == other.reset &&
      reliableFraction.bitPattern == other.reliableFraction.bitPattern &&
      warpedLumaError.bitPattern == other.warpedLumaError.bitPattern
  }
}

struct ErosionProbeSnapshot: Sendable {
  let preErode: Data
  fileprivate let production: ErosionProbeMotion
  fileprivate let frozen: ErosionProbeMotion
  let forward: ErosionProbeRawFlow
  let backward: ErosionProbeRawFlow
  let sourceIndex: Int
  let previousIndex: Int
  let randomAccessSubmitted: Bool

  init(production: MLXVideoMotion, frozen: FrozenErosionProbeVideoMotion,
    forward: MLXPixelBuffer, backward: MLXPixelBuffer, sourceIndex: Int,
    previousIndex: Int, randomAccessSubmitted: Bool) throws {
    let before = try XCTUnwrap(frozen.preErodeConfidence, "Frozen initializer did not retain evaluated quality")
    preErode = before.asData(access: .copy).data
    self.production = ErosionProbeMotion(vectors: production.vectors.asData(access: .copy).data,
      confidence: production.confidence.asData(access: .copy).data,
      reliableFraction: production.reliableFraction, warpedLumaError: production.warpedLumaError, reset: production.reset)
    self.frozen = ErosionProbeMotion(vectors: frozen.vectors.asData(access: .copy).data,
      confidence: frozen.confidence.asData(access: .copy).data,
      reliableFraction: frozen.reliableFraction, warpedLumaError: frozen.warpedLumaError, reset: frozen.reset)
    self.forward = try ErosionProbeRawFlow(forward)
    self.backward = try ErosionProbeRawFlow(backward)
    self.sourceIndex = sourceIndex; self.previousIndex = previousIndex
    self.randomAccessSubmitted = randomAccessSubmitted
  }
}

/// Measures only the existing erosion algorithm; opt-in, release, no model.
final class NativeMotionErosionCostTests: XCTestCase, @unchecked Sendable {
  private let sourceFlow = "vendor/MLX-DLSS/Sources/DLSSMedia/NativeOpticalFlow.swift"
  private let sourceMotion = "vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoMotion.swift"
  private let expectedFlow = "9b705fc307cc46dec9d50453a0066e57970c806e92ff5111d0aba4f465fc590d"
  private let expectedMotion = "d8878517b8e68ce8971c3c8e1813c92796303937ebde387cd875545b7b4e313e"
  private let inputSHA = "3c376d69b898a7fa0ee2474e233438c9071dbe3d63b3cece57d9b8e1bfa68181"
  private let width = 1920, height = 1080

  private struct Input: Decodable {
    struct Pin: Decodable { let path: String; let bytes: Int; let sha256: String }
    struct Time: Codable { let value: Int64; let timescale: Int32 }
    struct Frame: Decodable {
      let sourceFrameIndex: Int
      let pts: Time
      let duration: Time
      let path: String
      let sha256: String
      let bytes: Int
    }
    let width: Int
    let height: Int
    let frames: [Frame]
    let captureManifest: Pin
    let source: Pin
  }
  private struct Mask {
    let name: String
    let kind: String
    let sourceIndex: Int?
    let input: Data
    let expected: Data
    let inputFile: [String: Any]
    let expectedFile: [String: Any]
  }
  private func digest(_ bytes: Data) -> String {
    SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
  }
  private func fileDigest(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while let bytes = try file.read(upToCount: 1024 * 1024), !bytes.isEmpty { hash.update(data: bytes) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
  private func data(_ values: [Float]) -> Data { values.withUnsafeBytes { Data($0) } }
  private func summary(_ bytes: Data) -> [String: Any] {
    var nonfinite = 0, positive = 0, zero = 0, negative = 0
    var minimum = Float.infinity, maximum = -Float.infinity, sum = 0.0
    bytes.withUnsafeBytes { raw in
      for value in raw.bindMemory(to: Float.self) {
        if value > 0 { positive += 1 }
        else if value == 0 { zero += 1 }
        else if value < 0 { negative += 1 }
        if value.isFinite { minimum = min(minimum, value); maximum = max(maximum, value); sum += Double(value) }
        else { nonfinite += 1 }
      }
    }
    return ["bytes": bytes.count, "sha256": digest(bytes), "nonfiniteScalars": nonfinite,
      "positiveScalars": positive, "zeroScalars": zero, "negativeScalars": negative,
      "positiveFraction": Double(positive) / Double(max(1, bytes.count / 4)),
      "minimumFinite": minimum.isFinite ? Double(minimum) as Any : NSNull(),
      "maximumFinite": maximum.isFinite ? Double(maximum) as Any : NSNull(),
      "meanFinite": sum / Double(max(1, bytes.count / 4 - nonfinite))]
  }
  private func save(_ bytes: Data, name: String, output: URL) throws -> [String: Any] {
    try bytes.write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
    return ["path": name, "bytes": bytes.count, "sha256": digest(bytes)]
  }
  private func motionRecord(_ value: ErosionProbeMotion) -> [String: Any] {
    ["vectors": summary(value.vectors), "confidence": summary(value.confidence),
      "reliableFractionBits": value.reliableFraction.bitPattern,
      "warpedLumaErrorBits": value.warpedLumaError.bitPattern, "reset": value.reset]
  }
  private func flowRecord(_ value: ErosionProbeRawFlow, name: String, output: URL) throws -> [String: Any] {
    ["payload": try save(value.bytes, name: name, output: output),
      "width": value.width, "height": value.height, "pixelFormat": value.pixelFormat,
      "originalBytesPerRow": value.originalRowBytes, "logicalBytesPerRow": value.width * 4,
      "layout": "Tightly packed two-component Float16 native little-endian; original row padding excluded",
      "units": "flowPixels"]
  }
  private func reverse(_ source: String, renames: [(String, String)]) throws -> String {
    let marker = try NSRegularExpression(pattern: #"(?ms)^[ \t]*// EROSION-PROBE-BEGIN\n.*?^[ \t]*// EROSION-PROBE-END\n"#)
    var value = marker.stringByReplacingMatches(in: source,
      range: NSRange(source.startIndex..., in: source), withTemplate: "")
    for (original, renamed) in renames { value = value.replacingOccurrences(of: renamed, with: original) }
    return value
  }

  // A separate scope releases the VT session and retained source frames before timers.
  private func capture(input: Input, frames: [MLXVideoFrame], output: URL,
    record: (NativeMotionErosionCostTests.Mask?, [String: Any]) throws -> Void) async throws {
    let flow = try ErosionProbeNativeOpticalFlow(width: width, height: height, mode: .videoToolbox)
    try erosionRequire(flow.backend == "videotoolbox", "No fallback in actual-mask capture")
    var expectedRandom = true
    for (ordinal, source) in input.frames.enumerated() {
      let returned = try await flow.prepare(frames[ordinal], index: source.sourceFrameIndex, sceneCutThreshold: 0.3)
      let observed = await flow.takeSnapshot()
      var row: [String: Any] = ["ordinal": ordinal, "sourceFrameIndex": source.sourceFrameIndex,
        "pts": ["value": source.pts.value, "timescale": Int64(source.pts.timescale)],
        "duration": ["value": source.duration.value, "timescale": Int64(source.duration.timescale)],
        "inputSHA256": source.sha256, "hasMotion": returned != nil]
      if ordinal == 0 {
        try erosionRequire(returned == nil && observed == nil, "Cold frame must have no invented mask")
        row["preErodeMask"] = NSNull()
        try record(nil, row)
        continue
      }
      let snapshot = try XCTUnwrap(observed, "Missing capture from actual VT pair")
      let motion = try XCTUnwrap(returned)
      try erosionRequire(snapshot.sourceIndex == source.sourceFrameIndex && snapshot.previousIndex == input.frames[ordinal - 1].sourceFrameIndex &&
        snapshot.randomAccessSubmitted == expectedRandom, "History/random-access ownership changed")
      try erosionRequire(motion.reset == snapshot.production.reset &&
        motion.reliableFraction.bitPattern == snapshot.production.reliableFraction.bitPattern &&
        motion.warpedLumaError.bitPattern == snapshot.production.warpedLumaError.bitPattern,
        "Returned motion is not the production control")
      expectedRandom = motion.reset
      let name = "natural-\(source.sourceFrameIndex)"
      let beforeFile = try save(snapshot.preErode, name: "\(name)-pre-erode.f32", output: output)
      let afterFile = try save(snapshot.production.confidence, name: "\(name)-confidence.f32", output: output)
      row["preErodeMask"] = beforeFile; row["preErodeStatistics"] = summary(snapshot.preErode)
      row["productionMotion"] = motionRecord(snapshot.production); row["frozenMotion"] = motionRecord(snapshot.frozen)
      row["previousSourceFrameIndex"] = snapshot.previousIndex
      row["randomAccessSubmitted"] = snapshot.randomAccessSubmitted
      row["vtAPIIndexTimebase"] = 60
      row["sourcePTSIsDistinctFromVTIndexTimestamp"] = true
      row["forwardFlow"] = try flowRecord(snapshot.forward, name: "\(name)-forward.rg16f", output: output)
      row["backwardFlow"] = try flowRecord(snapshot.backward, name: "\(name)-backward.rg16f", output: output)
      let equal = snapshot.production.matches(snapshot.frozen)
      row["sameVTBuffersFullMotionExact"] = equal
      if !equal {
        row["mismatchProductionVectors"] = try save(snapshot.production.vectors, name: "\(name)-production-vectors.f32", output: output)
        row["mismatchFrozenVectors"] = try save(snapshot.frozen.vectors, name: "\(name)-frozen-vectors.f32", output: output)
        row["mismatchFrozenConfidence"] = try save(snapshot.frozen.confidence, name: "\(name)-frozen-confidence.f32", output: output)
      }
      let mask = Mask(name: name, kind: "actual-pre-erode", sourceIndex: source.sourceFrameIndex,
        input: snapshot.preErode, expected: snapshot.production.confidence,
        inputFile: beforeFile, expectedFile: afterFile)
      try record(mask, row)
      try erosionRequire(equal && snapshot.production.finite && snapshot.frozen.finite &&
        (summary(snapshot.preErode)["nonfiniteScalars"] as? Int) == 0,
        "Actual-mask capture/parity failed; mask/flow/mismatch payloads retained")
    }
  }

  private func synthetic(output: URL) throws -> [(Mask, [String: Any])] {
    let count = width * height
    let positive = (0..<count).map { Float(1 + $0 % 7) / 8 }
    var sparse = positive, expectedSparse = positive
    var faults = [(Int, Int)]()
    for y in stride(from: 0, to: height, by: 113) {
      for x in stride(from: 0, to: width, by: 97) { faults.append((x, y)) }
    }
    faults += [(width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    for (x, y) in faults {
      sparse[y * width + x] = 0
      // Every output within radius3 of a zero is invalid, including clamped borders.
      for yy in max(0, y - 3)...min(height - 1, y + 3) {
        for xx in max(0, x - 3)...min(width - 1, x + 3) { expectedSparse[yy * width + xx] = 0 }
      }
    }
    let dense = (0..<count).map { $0 % 4 == 0 ? Float(0.75) : Float(0) }
    let definitions: [(String, [Float], [Float], String)] = [
      ("all-positive", positive, positive, "Every center positive; preserve nonbinary original0.125...0.875 exactly"),
      ("sparse-invalid", sparse, expectedSparse, "Zero grid97x113 plus remaining corners; CPU expected zeros in radius3 of each fault"),
      ("dense-invalid", dense, [Float](repeating: 0, count: count), "Only linear indices divisible by4 are positive; every7x7 neighborhood contains zero"),
    ]
    return try definitions.map { name, values, expected, description in
      let bytes = data(values), outputBytes = data(expected)
      let a = try save(bytes, name: "synthetic-\(name)-input.f32", output: output)
      let b = try save(outputBytes, name: "synthetic-\(name)-expected.f32", output: output)
      return (Mask(name: name, kind: "synthetic", sourceIndex: nil, input: bytes,
        expected: outputBytes, inputFile: a, expectedFile: b),
        ["name": name, "definition": description, "input": a, "expected": b,
         "statistics": summary(bytes), "oracle": "Independent Boolean-predicate construction; no alternative GPU kernel"])
    }
  }

  func testCurrentErosionOnActualConfidence() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let outputPath = env["MLXDLSS_EROSION_COST_OUTPUT"] else { throw XCTSkip("Opt-in current erosion cost measurement") }
    #if DEBUG
    throw NSError(domain: "NativeMotionErosionCost", code: 2,
      userInfo: [NSLocalizedDescriptionKey: "Erosion measurement requires release configuration"])
    #endif
    let inputPath = try XCTUnwrap(env["MLXDLSS_EROSION_COST_INPUTS"])
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { root.deleteLastPathComponent() }
    let output = URL(fileURLWithPath: outputPath), inputURL = URL(fileURLWithPath: inputPath)
    try erosionRequire(!FileManager.default.fileExists(atPath: output.path) &&
      (try? output.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true, "Require new output path")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    var report: [String: Any] = ["schemaVersion": 1, "passed": false, "modelLoaded": false,
      "actualBackend": "videotoolbox", "completedPhase": "initializing", "progressUnits": 0,
      "expectedProgressUnits": 124, "actualMaskCount": 0, "syntheticMaskCount": 0,
      "mainErosionCalls": 0, "measuredErosionCalls": 0,
      "scope": "Current unchanged erosion cost on actual pre-erode masks; no optimization candidate",
      "width": width, "height": height, "erosionThreshold": "input>0.0f", "reliableFractionThreshold": "confidence>0.5",
      "timingScope": "Fresh original-erode output invocation plus blocking eval; inputs/params already materialized; readback and hashing afterward",
      "limitations": ["First-isolated is not cold or first compilation: frozen erode already runs in capture/parity.",
        "Pre-erode mask comes from source-identical test copy on same actual VT buffers, not access to private production storage.",
        "Isolated eval/input materialization/cache behavior does not reproduce upstream pipeline overlap; do not subtract from earlier bundled assessment time.",
        "Input/output readback and report writes outside timers still influence operating conditions.",
        "No model, candidate, playback, temporal-quality, Live, M5 or hard-memory claim."]]
    var captureFrames = [[String: Any]](), capturedMasks = [[String: Any]]()
    var samples = [[String: Any]](), syntheticControls = [[String: Any]]()
    var frozen = [String: String](), masks = [Mask]()
    var progress = 0
    func publish() throws {
      var value = report
      value["captureFrames"] = captureFrames; value["capturedMasks"] = capturedMasks
      value["samples"] = samples; value["syntheticControls"] = syntheticControls
      value["progressUnits"] = progress
      value["actualMaskCount"] = capturedMasks.count; value["syntheticMaskCount"] = syntheticControls.count
      value["mainErosionCalls"] = samples.count
      let measured = samples.filter { $0["measured"] as? Bool == true }
      value["measuredErosionCalls"] = measured.count
      value["measuredNaturalErosionCalls"] = measured.filter { $0["maskProvenance"] as? String == "natural" }.count
      value["measuredSyntheticErosionCalls"] = measured.filter { $0["maskProvenance"] as? String == "synthetic" }.count
      try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        .write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    let priorCache = MLXRuntimeDiagnostics.cacheLimitBytes
    report["priorMLXCachePolicyBytes"] = priorCache
    defer {
      do { try MLXRuntimeDiagnostics.setCacheLimitBytes(priorCache); report["restoredMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes }
      catch { report["passed"] = false; report["cacheRestorationError"] = error.localizedDescription }
      var after = [String: String]()
      for (path, expected) in frozen {
        do { after[path] = try fileDigest(URL(fileURLWithPath: path)) }
        catch { report["hashVerificationError"] = error.localizedDescription }
        if after[path] != expected { report["passed"] = false }
      }
      report["frozenSHA256After"] = after; report["frozenFilesUnchanged"] = !frozen.isEmpty && after == frozen
      try? publish()
    }
    do {
      if #available(macOS 15.4, *) { try erosionRequire(VTOpticalFlowConfiguration.isSupported, "Actual VT required") }
      else { try erosionRequire(false, "VT requires macOS15.4") }
      let flowCopy = "vendor/MLX-DLSS/Tests/DLSSMediaTests/ErosionProbeNativeOpticalFlow.swift"
      let motionCopy = "vendor/MLX-DLSS/Tests/DLSSMediaTests/FrozenErosionProbeVideoMotion.swift"
      let transformations: [(String, String, String, [(String, String)])] = [
        (sourceFlow, flowCopy, expectedFlow, [("NativeOpticalFlow", "ErosionProbeNativeOpticalFlow"),
          ("VideoToolboxFlowSession", "ErosionProbeVideoToolboxFlowSession")]),
        (sourceMotion, motionCopy, expectedMotion, [("MLXVideoMotion", "FrozenErosionProbeVideoMotion"),
          ("mlxdlss_native_flow_import", "mlxdlss_erosion_probe_flow_import"),
          ("mlxdlss_native_motion_quality", "mlxdlss_erosion_probe_motion_quality"),
          ("mlxdlss_native_motion_erode", "mlxdlss_erosion_probe_motion_erode"),
          ("mlxdlss_native_guide_resize", "mlxdlss_erosion_probe_guide_resize")]),
      ]
      var reversedPins = [String: String]()
      for (original, copy, expected, renames) in transformations {
        let bytes = try Data(contentsOf: root.appendingPathComponent(original))
        let renamed = try String(contentsOf: root.appendingPathComponent(copy), encoding: .utf8)
        let reversed = try reverse(renamed, renames: renames)
        try erosionRequire(digest(bytes) == expected && Data(reversed.utf8) == bytes, "Source reversal mismatch: \(copy)")
        reversedPins[original] = expected
      }
      report["reversedProductionSourceSHA256"] = reversedPins
      report["sourceCopyRule"] = "Remove complete EROSION-PROBE marker blocks, reverse explicit type/kernel rename lists; bytes must match entire production files"
      let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL)
      report["actualTestExecutable"] = executable.path
      let sourcePaths = [sourceFlow, sourceMotion, flowCopy, motionCopy,
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/NativeMotionErosionCostTests.swift",
        "vendor/MLX-DLSS/Sources/DLSSMedia/NativeImageIO.swift", "vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoFrame.swift",
        "vendor/MLX-DLSS/Sources/DLSSMLX/MLXRuntimeDiagnostics.swift",
        "vendor/MLX-DLSS/.build/release/mlx.metallib", ".build/debug/hdr-benchmark",
        ".build/debug/libFrameEngineShared.dylib", ".build/debug/mlx.metallib"]
      for path in sourcePaths { frozen[root.appendingPathComponent(path).path] = try fileDigest(root.appendingPathComponent(path)) }
      frozen[executable.path] = try fileDigest(executable)
      let inputBytes = try Data(contentsOf: inputURL)
      try erosionRequire(digest(inputBytes) == inputSHA, "Pinned12-frame input manifest differs")
      frozen[inputURL.path] = inputSHA
      report["inputManifest"] = ["path": inputURL.path, "bytes": inputBytes.count, "sha256": inputSHA]
      let input = try JSONDecoder().decode(Input.self, from: inputBytes)
      try erosionRequire(input.width == width && input.height == height && input.frames.map(\.sourceFrameIndex) == Array(1496...1507), "Input inventory differs")
      for entry in [input.captureManifest, input.source] {
        let url = entry.path.hasPrefix("/") ? URL(fileURLWithPath: entry.path) : root.appendingPathComponent(entry.path)
        try erosionRequire(fileDigest(url) == entry.sha256 && url.resourceValues(forKeys: [.fileSizeKey]).fileSize == entry.bytes, "Source/capture pin differs")
        frozen[url.path] = entry.sha256
      }
      try MLXRuntimeDiagnostics.setCacheLimitBytes(256 * 1024 * 1024)
      report["benchmarkMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
      var frames = [MLXVideoFrame]()
      for frame in input.frames {
        let url = frame.path.hasPrefix("/") ? URL(fileURLWithPath: frame.path) : root.appendingPathComponent(frame.path)
        let bytes = try Data(contentsOf: url)
        try erosionRequire(bytes.count == frame.bytes && bytes.count == width * height * 12 && digest(bytes) == frame.sha256,
          "Input proxy payload differs")
        try erosionRequire(frame.pts.timescale == 24000 && frame.duration.value == 1001 && frame.duration.timescale == 24000,
          "Exact input timing differs")
        frozen[url.path] = frame.sha256
        frames.append(try MLXVideoFrame(rgb: bytes, width: width, height: height))
      }
      report["frozenSHA256Before"] = frozen; report["completedPhase"] = "capture"
      try publish()
      try await capture(input: input, frames: frames, output: output) { mask, row in
        captureFrames.append(row)
        if let mask {
          masks.append(mask)
          capturedMasks.append(["name": mask.name, "sourceFrameIndex": mask.sourceIndex as Any? ?? NSNull(),
            "input": mask.inputFile, "expected": mask.expectedFile, "statistics": summary(mask.input)])
        }
        progress += 1
        try publish()
      }
      for (frame, source) in zip(frames, input.frames) {
        try erosionRequire(digest(frame.copyRGBData()) == source.sha256, "Retained source frame changed")
      }
      frames.removeAll()
      try erosionRequire(captureFrames.count == 12 && masks.count == 11, "Missing actual masks")
      for (mask, row) in try synthetic(output: output) { masks.append(mask); syntheticControls.append(row) }
      try erosionRequire(masks.count == 14, "Missing masks/controls")
      report["completedPhase"] = "isolated-erosion"; report["capturePhaseFinishedBeforeIsolatedTiming"] = true
      try publish()
      var evaluatedInputs = [MLXArray]()
      for mask in masks {
        try erosionRequire(mask.input.count == width * height * 4 && mask.expected.count == width * height * 4 &&
          (summary(mask.input)["nonfiniteScalars"] as? Int) == 0, "Invalid mask extent/domain")
        let input = MLXArray(mask.input, [1, height, width, 1], dtype: .float32)
        eval(input)
        try erosionRequire(input.asData(access: .copy).data == mask.input, "Input materialization changed bits")
        evaluatedInputs.append(input)
      }
      let params = MLXArray([UInt32(width), UInt32(height), 0, 0, 0, 0, 0, 0])
      eval(params)
      var allExact = true
      for pass in 0..<8 {
        let phase = pass == 0 ? "first-isolated" : pass < 4 ? "warmup" : "measured"
        let order = pass.isMultiple(of: 2) ? Array(masks.indices) : Array(masks.indices.reversed())
        for (orderIndex, index) in order.enumerated() {
          let mask = masks[index]
          let started = DispatchTime.now().uptimeNanoseconds
          let result = FrozenErosionProbeVideoMotion.originalErosion(evaluatedInputs[index], params: params)
          eval(result)
          let ended = DispatchTime.now().uptimeNanoseconds
          let elapsed = Double(ended - started) / 1e9
          let memory = MLXRuntimeDiagnostics.memorySnapshot()
          let bytes = result.asData(access: .copy).data
          let exact = bytes == mask.expected
          allExact = allExact && exact
          var row: [String: Any] = ["pass": pass, "phase": phase, "measured": pass >= 4,
            "orderIndex": orderIndex, "maskName": mask.name, "maskKind": mask.kind,
            "maskProvenance": mask.kind == "actual-pre-erode" ? "natural" : "synthetic",
            "sourceFrameIndex": mask.sourceIndex as Any? ?? NSNull(), "inputSHA256": digest(mask.input),
            "outputSHA256": digest(bytes), "expectedSHA256": digest(mask.expected), "byteEqual": exact,
            "startUptimeNanoseconds": started, "endUptimeNanoseconds": ended,
            "erosionCompletedWallSeconds": elapsed, "outputStatistics": summary(bytes),
            "mlxBytes": ["active": memory.activeBytes, "cache": memory.cacheBytes, "peakActive": memory.peakActiveBytes]]
          if !exact { row["mismatchPayload"] = try save(bytes, name: "mismatch-\(pass)-\(mask.name).f32", output: output) }
          samples.append(row); progress += 1
          try publish()
          try erosionRequire((summary(bytes)["nonfiniteScalars"] as? Int) == 0, "Nonfinite erosion output")
        }
      }
      for (mask, input) in zip(masks, evaluatedInputs) {
        try erosionRequire(input.asData(access: .copy).data == mask.input, "Retained mask changed during timing")
        for entry in [mask.inputFile, mask.expectedFile] {
          let name = try XCTUnwrap(entry["path"] as? String), expected = try XCTUnwrap(entry["sha256"] as? String)
          try erosionRequire(fileDigest(output.appendingPathComponent(name)) == expected, "Retained mask/reference file changed")
        }
      }
      for row in captureFrames where row["hasMotion"] as? Bool == true {
        for direction in ["forwardFlow", "backwardFlow"] {
          let flow = try XCTUnwrap(row[direction] as? [String: Any])
          let entry = try XCTUnwrap(flow["payload"] as? [String: Any])
          let name = try XCTUnwrap(entry["path"] as? String), expected = try XCTUnwrap(entry["sha256"] as? String)
          let bytes = try XCTUnwrap(entry["bytes"] as? Int), url = output.appendingPathComponent(name)
          try erosionRequire(fileDigest(url) == expected && url.resourceValues(forKeys: [.fileSizeKey]).fileSize == bytes,
            "Retained raw VT flow changed")
        }
      }
      for (path, expected) in frozen { try erosionRequire(fileDigest(URL(fileURLWithPath: path)) == expected, "Frozen file changed") }
      report["allErosionOutputsExact"] = allExact
      report["allInputFramesAndMasksUnchanged"] = true
      report["completedPhase"] = "complete"
      report["passed"] = allExact && progress == 124 && samples.count == 112 &&
        samples.filter { $0["measured"] as? Bool == true }.count == 56 && syntheticControls.count == 3
      try publish()
      try erosionRequire(report["passed"] as? Bool == true, "Current erosion measurement integrity failed")
    } catch {
      report["failure"] = error.localizedDescription
      try? publish()
      throw error
    }
  }
}
