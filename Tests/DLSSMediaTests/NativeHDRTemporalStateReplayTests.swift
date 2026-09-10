#if MLXDLSS_TEMPORAL_DIAGNOSTICS
import CoreMedia
import CryptoKit
import Foundation
import MLX
import XCTest
@testable import DLSSMedia
@testable import DLSSMLX

/// Explicitly opted-in diagnostic: 48 unchanged native frames, then four
/// noncommitting state replays. No test exists without the compilation flag.
final class NativeHDRTemporalStateReplayTests: XCTestCase, @unchecked Sendable {
  private let inputSHA = "08790dec58ec181e927072400d9defcffd6b974b26e51b816d9558a2cdbb6262"
  private let captureSHA = "fa1747a8432fb229e7890dea060f7486b7e679909411500bca7cc73e49f8c1fa"
  private let width = 960, height = 540
  private let maximumDumpBytes = 1024 * 1024 * 1024

  private struct Time: Codable, Equatable {
    let value: Int64
    let timescale: Int32
    var cm: CMTime { CMTime(value: value, timescale: timescale) }
    var json: [String: Any] { ["value": value, "timescale": timescale] }
  }
  private struct Pin: Decodable {
    let path: String
    let bytes: Int
    let sha256: String
  }
  private struct Input: Decodable {
    struct Frame: Decodable {
      let sourceFrameIndex: UInt64
      let path: String
      let bytes: Int
      let sha256: String
      let pts: Time
      let duration: Time
    }
    let schemaVersion, width, height: Int
    let layout, primaries, transfer, units: String
    let frames: [Frame]
  }
  private struct Capture: Decodable {
    struct Frame: Decodable {
      let ordinal: Int
      let sourceFrameIndex: UInt64
      let inputPath, inputSHA256: String
      let pts: Time
      let duration: Time
      let generation: UInt64
      let usedModel, historyReset: Bool
      let knownInputDiscontinuities: [String]
      let views: [String: Pin]
    }
    struct Model: Decodable { let path: String; let files: [String: String] }
    let complete: Bool
    let schemaVersion, width, height, completedFrames, requestedFrames: Int
    let layout, inputManifestCopy, inputManifestSHA256, sourceIdentity: String
    let generation: UInt64
    let views: [String]
    let frames: [Frame]
    let model: Model
  }
  private struct RetainedFrame {
    let source: MLXHDRFrame
    let proxy: MLXVideoFrame
    let enhanced: Data
  }

  private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw MLXMediaError(message) }
  }
  private func digest(_ bytes: Data) -> String {
    SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
  }
  private func fileDigest(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hash = SHA256()
    while let bytes = try handle.read(upToCount: 1024 * 1024), !bytes.isEmpty { hash.update(data: bytes) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }
  private func filePin(_ url: URL) throws -> [String: Any] {
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    try require(values.isRegularFile == true, "Not a regular file: \(url.path)")
    return ["path": url.path, "resolvedPath": url.resolvingSymlinksInPath().path,
      "bytes": try XCTUnwrap(values.fileSize), "sha256": try fileDigest(url)]
  }
  private func contained(_ path: String, under directory: URL) throws -> URL {
    try require(!path.isEmpty && !(path as NSString).isAbsolutePath &&
      !path.split(separator: "/").contains(".."), "Expected contained relative path")
    let base = directory.standardizedFileURL.resolvingSymlinksInPath()
    let value = base.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
    try require(value.path.hasPrefix(base.path + "/"), "Path escapes input directory")
    return value
  }
  private func json(_ bytes: Data) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
  }
  private func publish(_ report: [String: Any], to output: URL) throws {
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try (data + Data([10])).write(to: output.appendingPathComponent("report.json"), options: .atomic)
  }
  private func memory() -> [String: Any] {
    let value = MLXRuntimeDiagnostics.memorySnapshot()
    return ["activeBytes": value.activeBytes, "cacheBytes": value.cacheBytes,
      "peakActiveBytes": value.peakActiveBytes]
  }
  private func scalarValues(_ bytes: Data, dtype: String) throws -> [Double]? {
    switch dtype {
    case "float16": return bytes.withUnsafeBytes { $0.bindMemory(to: Float16.self).map(Double.init) }
    case "float32": return bytes.withUnsafeBytes { $0.bindMemory(to: Float.self).map(Double.init) }
    case "float64": return bytes.withUnsafeBytes { Array($0.bindMemory(to: Double.self)) }
    case "int8", "uint8", "bool", "int16", "uint16", "int32", "uint32", "int64", "uint64": return nil
    default: throw MLXMediaError("Unrecognized captured native dtype: \(dtype)")
    }
  }
  private func tensorMetadata(_ tensor: MLXTemporalDiagnosticTensor) throws -> [String: Any] {
    let sizes = ["float16": 2, "float32": 4, "float64": 8, "int8": 1, "uint8": 1,
      "bool": 1, "int16": 2, "uint16": 2, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8]
    let size = try XCTUnwrap(sizes[tensor.dtype], "Unsupported native tensor dtype")
    var elements = 1
    for dimension in tensor.shape {
      try require(dimension > 0 && elements <= maximumDumpBytes / dimension, "Tensor extent exceeds diagnostic bound")
      elements *= dimension
    }
    try require(elements <= maximumDumpBytes / size && tensor.bytes.count == elements * size,
      "Native tensor dtype/shape/byte count mismatch")
    var result: [String: Any] = ["shape": tensor.shape, "dtype": tensor.dtype,
      "elements": elements, "bytes": tensor.bytes.count, "sha256": digest(tensor.bytes),
      "layout": "Contiguous native dtype, native little-endian; no float32 promotion in stored bytes"]
    if let values = try scalarValues(tensor.bytes, dtype: tensor.dtype) {
      let nonfinite = values.filter { !$0.isFinite }.count
      result["nonfiniteScalars"] = nonfinite
      result["minimum"] = values.filter(\.isFinite).min() as Any? ?? NSNull()
      result["maximum"] = values.filter(\.isFinite).max() as Any? ?? NSNull()
      try require(nonfinite == 0, "Nonfinite diagnostic tensor; raw bytes are retained")
    } else {
      result["nonfiniteScalars"] = NSNull()
      result["numericScope"] = "Integer/index tensor; raw bytes and native dtype are authoritative"
    }
    return result
  }
  private func difference(_ actual: Data, _ expected: Data, dtype: String) throws -> [String: Any] {
    let equal = actual == expected
    var result: [String: Any] = ["bytesEqual": equal, "actualSHA256": digest(actual),
      "expectedSHA256": digest(expected), "actualBytes": actual.count, "expectedBytes": expected.count]
    guard actual.count == expected.count else {
      result["numericDifference"] = NSNull(); result["reason"] = "Different byte extents"; return result
    }
    if equal {
      result["differentBytes"] = 0
      result["numericDifference"] = ["maximumAbsolute": 0.0, "meanAbsolute": 0.0, "rms": 0.0, "signedMean": 0.0]
      return result
    }
    result["differentBytes"] = actual.withUnsafeBytes { left in
      expected.withUnsafeBytes { right in
        let a = left.bindMemory(to: UInt8.self), b = right.bindMemory(to: UInt8.self)
        var count = 0
        for index in a.indices where a[index] != b[index] { count += 1 }
        return count
      }
    }
    if let a = try scalarValues(actual, dtype: dtype), let b = try scalarValues(expected, dtype: dtype) {
      try require(a.count == b.count, "Comparison scalar extents differ")
      let nonfiniteA = a.filter { !$0.isFinite }.count, nonfiniteB = b.filter { !$0.isFinite }.count
      if nonfiniteA != 0 || nonfiniteB != 0 {
        result["numericDifference"] = NSNull(); result["reason"] = "Nonfinite comparison inputs"
        result["actualNonfiniteScalars"] = nonfiniteA; result["expectedNonfiniteScalars"] = nonfiniteB
        return result
      }
      var absolute = 0.0, square = 0.0, signed = 0.0, maximum = 0.0
      for (left, right) in zip(a, b) {
        let delta = left - right
        absolute += abs(delta); square += delta * delta; signed += delta; maximum = max(maximum, abs(delta))
      }
      let count = Double(max(1, a.count))
      result["numericDifference"] = ["maximumAbsolute": maximum, "meanAbsolute": absolute / count,
        "rms": sqrt(square / count), "signedMean": signed / count]
    } else {
      result["numericDifference"] = NSNull(); result["reason"] = "Integer/index tensor; compare exact bytes"
    }
    return result
  }
  private func compareTensors(_ actual: [String: MLXTemporalDiagnosticTensor],
    _ expected: [String: MLXTemporalDiagnosticTensor]) throws -> [String: Any] {
    var comparisons = [String: Any]()
    for name in Set(actual.keys).union(expected.keys).sorted() {
      guard let a = actual[name], let b = expected[name] else {
        comparisons[name] = ["comparable": false, "reason": "Tensor is absent in one snapshot"]
        continue
      }
      if a.shape != b.shape || a.dtype != b.dtype {
        comparisons[name] = ["comparable": false, "reason": "Shape or native dtype differs",
          "actualShape": a.shape, "expectedShape": b.shape, "actualDType": a.dtype, "expectedDType": b.dtype]
      } else {
        comparisons[name] = try difference(a.bytes, b.bytes, dtype: a.dtype)
      }
    }
    return ["allNativeTensorsExactlyEqual": actual == expected, "tensors": comparisons]
  }

  func testCapturedOcclusionStateReplay() async throws {
    let environment = ProcessInfo.processInfo.environment
    let names = ["MLXDLSS_TEMPORAL_STATE_INPUT", "MLXDLSS_TEMPORAL_STATE_CAPTURE",
      "MLXDLSS_TEMPORAL_STATE_MODEL", "MLXDLSS_TEMPORAL_STATE_OUTPUT"]
    guard names.contains(where: { environment[$0] != nil }) else {
      throw XCTSkip("Set all four MLXDLSS_TEMPORAL_STATE input/capture/model/output paths to opt in")
    }
    let arguments = try names.map { try XCTUnwrap(environment[$0], "Missing explicit \($0)") }
    #if DEBUG
    throw XCTSkip("The temporal state replay is an opt-in release diagnostic")
    #endif
    try require(UInt16(1).littleEndian == 1, "Native diagnostic storage requires little-endian host")
    let inputURL = URL(fileURLWithPath: arguments[0]).standardizedFileURL.resolvingSymlinksInPath()
    let captureURL = URL(fileURLWithPath: arguments[1]).standardizedFileURL.resolvingSymlinksInPath()
    let modelURL = URL(fileURLWithPath: arguments[2]).standardizedFileURL.resolvingSymlinksInPath()
    let output = URL(fileURLWithPath: arguments[3]).standardizedFileURL
    let manager = FileManager.default
    try require(!manager.fileExists(atPath: output.path) &&
      (try? manager.destinationOfSymbolicLink(atPath: output.path)) == nil,
      "Output exists, including dangling symlink; use a new directory")
    try manager.createDirectory(at: output, withIntermediateDirectories: false)
    var report: [String: Any] = ["schemaVersion": 1, "complete": false, "passed": false,
      "completedUnits": 0, "expectedUnits": 52, "completedPhase": "preflight",
      "baselineFrames": [[String: Any]](), "replayCases": [[String: Any]](),
      "modelCallsAttempted": 0, "completedModelCalls": 0, "baselineViewComparisons": 0,
      "compilationFlag": "MLXDLSS_TEMPORAL_DIAGNOSTICS", "environmentPaths": Dictionary(uniqueKeysWithValues: zip(names, arguments)),
      "scope": "Isolated diagnostic only.48 unchanged native frames must reproduce all192 prior views before four noncommitting private-state replays. No player, shared ABI or source-rate/physical presentation qualification.",
      "limitations": ["Exact reproduction is an admission gate for this diagnostic, not a quality claim.",
        "Base45 color/motion/confidence remain fixed while captured noise and incoming history vary.",
        "Comparisons to actual46 are separate: its guides/inputs may differ and are retained rather than assumed equal.",
        "Completed wall times include diagnostics and readback; they are not performance measurements.",
        "A256MiB soft cache policy is not an instantaneous allocation or process-memory limit."]]
    var pins = [String: [String: Any]](), baselineRows = [[String: Any]](), replayRows = [[String: Any]]()
    var savedPayloads = [[String: Any]]()
    var dumpedBytes = 0, completed = 0, attempted = 0, comparedViews = 0
    var processor: NativeHDRProcessor?
    let start = DispatchTime.now().uptimeNanoseconds
    func publish() throws {
      report["baselineFrames"] = baselineRows; report["replayCases"] = replayRows
      report["completedUnits"] = completed; report["completedModelCalls"] = completed
      report["modelCallsAttempted"] = attempted; report["baselineViewComparisons"] = comparedViews
      report["dumpedPayloadBytes"] = dumpedBytes
      report["savedPayloads"] = savedPayloads
      report["elapsedSeconds"] = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
      try self.publish(report, to: output)
    }
    func remember(_ url: URL, expected: String? = nil) throws {
      let value = try filePin(url)
      if let expected { try require(value["sha256"] as? String == expected, "File hash differs: \(url.path)") }
      if let previous = pins[url.path] {
        try require(NSDictionary(dictionary: previous).isEqual(to: value), "Pinned file changed: \(url.path)")
      }
      pins[url.path] = value
    }
    func read(_ url: URL, count: Int? = nil, sha: String? = nil) throws -> Data {
      let bytes = try Data(contentsOf: url)
      if let count { try require(bytes.count == count, "Byte count differs: \(url.path)") }
      if let sha { try require(digest(bytes) == sha, "Payload hash differs: \(url.path)") }
      try remember(url, expected: sha)
      return bytes
    }
    func save(_ bytes: Data, name: String) throws -> [String: Any] {
      try require(bytes.count <= maximumDumpBytes - dumpedBytes, "Diagnostic dump exceeds1GiB bound")
      try bytes.write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
      dumpedBytes += bytes.count
      let pin: [String: Any] = ["path": name, "bytes": bytes.count, "sha256": digest(bytes)]
      savedPayloads.append(pin)
      return pin
    }
    func dumpTensor(_ tensor: MLXTemporalDiagnosticTensor, name: String) throws -> [String: Any] {
      let payload = try save(tensor.bytes, name: name + ".bin")
      var value = try tensorMetadata(tensor)
      value["payload"] = payload
      return value
    }
    func dumpSnapshot(_ value: MLXTemporalDiagnosticSnapshot, prefix: String) throws -> [String: Any] {
      var tensors = [String: Any]()
      for name in value.tensors.keys.sorted() { tensors[name] = try dumpTensor(value.tensors[name]!, name: prefix + "-" + name) }
      return ["frameIndex": value.frameIndex, "noiseFrameIndex": value.noiseFrameIndex,
        "logicalWidth": value.logicalWidth, "logicalHeight": value.logicalHeight,
        "networkWidth": value.networkWidth, "networkHeight": value.networkHeight, "tensors": tensors]
    }
    func dumpState(_ value: MLXTemporalDiagnosticState, prefix: String) throws -> [String: Any] {
      var tensors = [String: Any]()
      for name in value.tensors.keys.sorted() { tensors[name] = try dumpTensor(value.tensors[name]!, name: prefix + "-" + name) }
      return ["noiseFrameIndex": value.noiseFrameIndex, "lifecycleDescription": value.lifecycleDescription,
        "nativeCompositionDescription": value.nativeCompositionDescription,
        "deviceFeaturesEnabled": value.deviceFeaturesEnabled, "tensors": tensors]
    }
    try publish()
    do {
      let inputBytes = try read(inputURL, sha: inputSHA)
      let captureBytes = try read(captureURL, sha: captureSHA)
      let decoder = JSONDecoder()
      let input = try decoder.decode(Input.self, from: inputBytes)
      let capture = try decoder.decode(Capture.self, from: captureBytes)
      let inputJSON = try json(inputBytes), captureJSON = try json(captureBytes)
      let viewNames = ["original", "proxy", "identity", "enhanced"]
      try require(input.schemaVersion == 1 && capture.schemaVersion == 1 && capture.complete &&
        input.frames.count == 48 && capture.frames.count == 48 && capture.completedFrames == 48 && capture.requestedFrames == 48 &&
        input.width == width && input.height == height && capture.width == width && capture.height == height &&
        input.layout == "RGB float32 little-endian top-to-bottom" && input.layout == capture.layout &&
        input.primaries == "BT.2020" && input.transfer == "linear" && input.units == "cd/m2" &&
        capture.views == viewNames && capture.inputManifestSHA256 == inputSHA &&
        capture.sourceIdentity == "sha256:" + inputSHA && capture.generation == 1, "Canonical capture contract differs")
      let copiedURL = try contained(capture.inputManifestCopy, under: captureURL.deletingLastPathComponent())
      try require(try read(copiedURL, sha: inputSHA) == inputBytes, "Capture input copy differs")
      let settings = try XCTUnwrap(captureJSON["settings"] as? [String: Any])
      let expectedSettings: [String: Any] = ["processingWidth": 512, "processingHeight": 288,
        "strength": 1, "colourStrength": 1, "maximumLuminanceRatio": 2, "referenceWhiteNits": 203,
        "temporal": true, "motionRequested": "automatic", "sceneCutThreshold": 0.3,
        "modelInputRange": "bounded-sRGB-after-resample", "precision": "float16", "mlxCacheBytes": 268435456]
      try require(NSDictionary(dictionary: settings).isEqual(to: expectedSettings), "Prior capture settings differ")
      report["settings"] = settings; report["executionMode"] = "metalFused"
      report["priorCaptureMetadata"] = captureJSON.filter { $0.key != "frames" }
      report["inputManifest"] = pins[inputURL.path]; report["priorCaptureManifest"] = pins[captureURL.path]
      report["sourceProvenance"] = inputJSON["provenance"]
      try require(Set(capture.model.files.keys) == Set(["manifest.json", "weights.safetensors"]), "Unexpected model file inventory")
      for name in capture.model.files.keys.sorted() {
        try remember(try contained(name, under: modelURL), expected: capture.model.files[name])
      }
      let checkout = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
      for path in ["Tests/DLSSMediaTests/NativeHDRTemporalStateReplayTests.swift",
        "Sources/DLSSMLX/MLXTemporalDiagnostics.swift", "Sources/DLSSMLX/MLXNeuralRenderingDeviceTemporalBackend.swift",
        "Sources/DLSSMedia/NativeHDRProcessor.swift", "Sources/DLSSMLX/MLXNeuralRenderingDisplayCodec.swift"] {
        try remember(checkout.appendingPathComponent(path))
      }
      let executable = try XCTUnwrap(Bundle(for: NativeHDRTemporalStateReplayTests.self).executableURL)
      let metallib = executable.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
      try remember(executable); try remember(metallib)
      report["actualTestExecutable"] = executable.path; report["actualTestBundleMetallib"] = metallib.path
      for index in input.frames.indices {
        let source = input.frames[index], prior = capture.frames[index]
        try require(source.sourceFrameIndex == UInt64(index) && prior.sourceFrameIndex == UInt64(index) && prior.ordinal == index &&
          source.pts == Time(value: Int64(index), timescale: 30) && prior.pts == source.pts &&
          source.duration == Time(value: 1, timescale: 30) && prior.duration == source.duration &&
          prior.generation == capture.generation && prior.inputSHA256 == source.sha256 && prior.inputPath == source.path &&
          prior.usedModel && prior.historyReset == (index == 0) && Set(prior.views.keys) == Set(viewNames), "Frame pairing/timing/reset differs at\(index)")
        let raw = try read(try contained(source.path, under: inputURL.deletingLastPathComponent()), count: width * height * 12, sha: source.sha256)
        try require(raw.withUnsafeBytes { $0.bindMemory(to: Float.self).allSatisfy { $0.isFinite && $0 > 0 } }, "Nonfinite/nonpositive source")
        for name in viewNames {
          let pin = prior.views[name]!
          try require(pin.bytes == width * height * 12, "Prior view extent differs")
          let bytes = try read(try contained(pin.path, under: captureURL.deletingLastPathComponent()), count: pin.bytes, sha: pin.sha256)
          try require(bytes.withUnsafeBytes { $0.bindMemory(to: Float.self).allSatisfy(\.isFinite) }, "Nonfinite prior view")
          if name == "original" { try require(bytes == raw, "Prior original differs from source") }
        }
      }
      report["frozenSHA256Before"] = pins.values.sorted { ($0["path"] as! String) < ($1["path"] as! String) }
      try MLXRuntimeDiagnostics.setCacheLimitBytes(256 * 1024 * 1024)
      MLXRuntimeDiagnostics.resetPeakMemory()
      report["cachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
      report["cacheScope"] = "Standalone filtered process;256MiB soft free-cache policy set and peak reset before model initialization. No parent-process cache restoration is claimed."
      processor = try NativeHDRProcessor(configuration: .init(modelURL: modelURL,
        processingWidth: 512, processingHeight: 288, strength: 1, colorStrength: 1,
        maximumLuminanceRatio: 2, temporal: true, motion: .automatic, precision: .float16, sceneCutThreshold: 0.3))
      let native = try XCTUnwrap(processor)
      try await native.enableTemporalDiagnostics(frameIndices: [45, 46])
      var retained = [UInt64: RetainedFrame]()
      report["completedPhase"] = "baseline"
      for index in input.frames.indices {
        let source = input.frames[index], prior = capture.frames[index]
        report["currentSourceFrameIndex"] = index
        try publish()
        let raw = try read(try contained(source.path, under: inputURL.deletingLastPathComponent()), count: source.bytes, sha: source.sha256)
        let original = try MLXVideoFrame(rgb: raw, width: width, height: height)
        let metadata = MLXHDRFrameMetadata(time: source.pts.cm, duration: source.duration.cm,
          sourceID: capture.sourceIdentity, streamID: 1, frameIndex: source.sourceFrameIndex,
          generation: capture.generation, crop: CGRect(x: 0, y: 0, width: width, height: height),
          color: .init(transfer: .linear, primaries: .bt2020, fullRange: true, referenceWhiteNits: 203,
            sourceTags: ["rawDomain": "linearBT2020nits", "matrix": "not used for direct RGB", "provenance": "input-manifest.json"]))
        let frame = MLXHDRFrame(original: original, metadata: metadata)
        attempted += 1; try publish()
        let began = DispatchTime.now().uptimeNanoseconds
        let result = try await native.process(frame)
        let ended = DispatchTime.now().uptimeNanoseconds
        var comparisons = [String: Any](), allEqual = true
        var enhancedBytes = Data()
        for (name, view) in [("original", result.original), ("proxy", result.proxy), ("identity", result.identity), ("enhanced", result.enhanced)] {
          let actual = view.copyRGBData(), pin = prior.views[name]!
          let expected = try read(try contained(pin.path, under: captureURL.deletingLastPathComponent()), count: pin.bytes, sha: pin.sha256)
          let finite = actual.withUnsafeBytes { $0.bindMemory(to: Float.self).allSatisfy(\.isFinite) }
          let equal = view.width == width && view.height == height && actual == expected && finite
          var failedPayload: [String: Any]?
          if !equal { failedPayload = try save(actual, name: "failure-frame-\(index)-\(name).rgb32f") }
          var comparison = try difference(actual, expected, dtype: "float32")
          comparison["finite"] = finite; comparison["referencePath"] = pin.path
          if let failedPayload { comparison["failureActualPayload"] = failedPayload }
          comparisons[name] = comparison; allEqual = allEqual && equal; comparedViews += 1
          if name == "enhanced" { enhancedBytes = actual }
        }
        let provenanceEqual = result.metadata.time == metadata.time && result.metadata.duration == metadata.duration &&
          result.metadata.frameIndex == metadata.frameIndex && result.metadata.sourceID == metadata.sourceID &&
          result.metadata.streamID == metadata.streamID && result.metadata.generation == metadata.generation &&
          result.usedModel && result.historyReset == prior.historyReset
        completed += 1
        baselineRows.append(["sourceFrameIndex": source.sourceFrameIndex, "pts": source.pts.json,
          "duration": source.duration.json, "inputSHA256": source.sha256, "generation": metadata.generation,
          "historyReset": result.historyReset, "usedModel": result.usedModel,
          "provenanceEqual": provenanceEqual, "allFourViewsExact": allEqual, "views": comparisons,
          "startedUptimeNanoseconds": began, "endedUptimeNanoseconds": ended,
          "processCompletedWallSeconds": Double(ended - began) / 1e9, "mlxBytes": memory()])
        if index == 45 || index == 46 { retained[source.sourceFrameIndex] = RetainedFrame(source: frame, proxy: result.proxy, enhanced: enhancedBytes) }
        try publish()
        try require(allEqual && provenanceEqual, "Baseline parity failed at\(index); failure payload/comparisons retained and no ablation allowed")
      }
      try require(completed == 48 && comparedViews == 192, "Incomplete unchanged baseline")
      report["baselineParityPassed"] = true; report["completedPhase"] = "export-states"
      try publish()
      // First diagnostic state readbacks occur only after all baseline exports.
      let captured = try await native.temporalDiagnosticSnapshots()
      try require(Set(captured.map(\.frameIndex)) == Set([UInt64(45), UInt64(46)]) && captured.count == 2, "Missing or extra captured states")
      let snapshots = Dictionary(uniqueKeysWithValues: captured.map { ($0.frameIndex, $0) })
      var snapshotFiles = [String: Any]()
      for index in [UInt64(45), UInt64(46)] {
        let snapshot = snapshots[index]!, kept = retained[index]!
        snapshotFiles[String(index)] = try dumpSnapshot(snapshot, prefix: "captured-\(index)")
        try require(snapshot.logicalWidth == 512 && snapshot.logicalHeight == 288 && snapshot.networkWidth == 512 && snapshot.networkHeight == 320,
          "Unexpected logical/padded network extent")
        let required = Set(["color", "incomingHistory", "motion", "confidence", "depth", "logicalFeatures", "networkFeatures",
          "preparedFeatures", "networkHead", "logicalHead", "postprocessedSDR", "fullResolutionProxy", "fullResolutionModel"])
        try require(required.isSubset(of: Set(snapshot.tensors.keys)), "Captured state lacks a required actual intermediate")
        try require(snapshot.tensors["fullResolutionProxy"]!.bytes == kept.proxy.copyRGBData(), "Captured proxy differs from returned baseline")
        try require(snapshot.tensors["fullResolutionModel"]!.dtype == "float32" &&
          snapshot.tensors["fullResolutionModel"]!.shape == [1, height, width, 3], "Model is not full-resolution nativeFloat32 RGB")
      }
      report["snapshots"] = snapshotFiles
      let base = snapshots[45]!, actual46 = snapshots[46]!
      report["actual45Versus46Intermediates"] = try compareTensors(actual46.tensors, base.tensors)
      let matrixBefore = try await native.temporalDiagnosticState()
      report["matrixStateBefore"] = try dumpState(matrixBefore, prefix: "matrix-before")
      report["completedPhase"] = "noncommitting-replays"
      try publish()
      let codec = MLXNeuralRenderingDisplayCodec()
      let kept45 = retained[45]!, kept46 = retained[46]!
      let combinations: [(UInt64, UInt64)] = [(45, 45), (46, 45), (45, 46), (46, 46)]
      for (ordinal, selectors) in combinations.enumerated() {
        let (noiseFrom, historyFrom) = selectors
        report["currentReplayOrdinal"] = ordinal; attempted += 1; try publish()
        let began = DispatchTime.now().uptimeNanoseconds
        let replay = try await native.replayTemporalDiagnostic(baseFrameIndex: 45,
          noiseFromFrameIndex: noiseFrom, historyFromFrameIndex: historyFrom)
        let enhanced = codec.resolve(proxy: kept45.proxy, model: replay.fullResolutionModel, original: kept45.source.original,
          configuration: .init(whitePoint: 203, transferStrength: 1, colorStrength: 1,
            maximumLuminanceRatio: 2, workingPrimaries: .bt2020))
        let enhancedData = enhanced.copyRGBData()
        let ended = DispatchTime.now().uptimeNanoseconds
        completed += 1
        let prefix = "replay-\(ordinal)-noise\(noiseFrom)-history\(historyFrom)"
        var row: [String: Any] = ["ordinal": ordinal, "baseSourceFrameIndex": 45,
          "noiseFromFrameIndex": noiseFrom, "historyFromFrameIndex": historyFrom,
          "incomingHistoryProducedByFrame": historyFrom - 1,
          "snapshot": try dumpSnapshot(replay.snapshot, prefix: prefix),
          "enhanced": try save(enhancedData, name: prefix + "-enhanced.rgb32f"),
          "stateBefore": try dumpState(replay.stateBefore, prefix: prefix + "-state-before"),
          "stateAfter": try dumpState(replay.stateAfter, prefix: prefix + "-state-after"),
          "stateUnchanged": replay.stateUnchanged,
          "stateEqualsMatrixInitial": replay.stateBefore == matrixBefore && replay.stateAfter == matrixBefore,
          "versusCaptured45Intermediates": try compareTensors(replay.snapshot.tensors, base.tensors),
          "versusActual46Intermediates": try compareTensors(replay.snapshot.tensors, actual46.tensors),
          "enhancedVersusCaptured45": try difference(enhancedData, kept45.enhanced, dtype: "float32"),
          "enhancedVersusActual46": try difference(enhancedData, kept46.enhanced, dtype: "float32"),
          "startedUptimeNanoseconds": began, "endedUptimeNanoseconds": ended,
          "completedModelAndResolveWallSeconds": Double(ended - began) / 1e9, "mlxBytes": memory()]
        let returnedModel = replay.fullResolutionModel.copyRGBData()
        let exactModelExport = replay.snapshot.tensors["fullResolutionModel"]?.bytes == returnedModel
        let fixedInputsExact = ["color", "motion", "confidence", "depth", "controlMask", "fullResolutionProxy"].allSatisfy {
          replay.snapshot.tensors[$0] == base.tensors[$0]
        }
        row["returnedModelMatchesSnapshotBytes"] = exactModelExport
        row["fixedBase45InputsExact"] = fixedInputsExact
        let originalCombinationExact = replay.snapshot == base && enhancedData == kept45.enhanced
        row["originalCombinationExact"] = ordinal == 0 ? originalCombinationExact as Any : NSNull()
        replayRows.append(row); try publish()
        try require(exactModelExport && fixedInputsExact && replay.stateUnchanged && replay.stateBefore == matrixBefore && replay.stateAfter == matrixBefore,
          "Replay committed/mutated retained state or exported a different model; retain failure")
        try require(replay.snapshot.frameIndex == 45 && replay.snapshot.noiseFrameIndex == snapshots[noiseFrom]!.noiseFrameIndex &&
          replay.snapshot.tensors["incomingHistory"] == snapshots[historyFrom]!.tensors["incomingHistory"], "Replay selectors differ from captured state")
        try require(enhancedData.withUnsafeBytes { $0.bindMemory(to: Float.self).allSatisfy(\.isFinite) }, "Nonfinite replay enhanced output")
        if ordinal == 0 { try require(originalCombinationExact, "Original state/noise replay is not byte exact; no later combinations allowed") }
      }
      let matrixAfter = try await native.temporalDiagnosticState()
      report["matrixStateAfter"] = try dumpState(matrixAfter, prefix: "matrix-after")
      try require(matrixAfter == matrixBefore && completed == 52 && attempted == 52 && replayRows.count == 4, "Incomplete or committing replay matrix")
      report["matrixStateUnchanged"] = true; report["completedPhase"] = "verify-pins"
      try publish()
      for value in Array(pins.values) {
        let url = URL(fileURLWithPath: value["path"] as! String)
        try remember(url, expected: value["sha256"] as? String)
      }
      for value in savedPayloads {
        let url = output.appendingPathComponent(value["path"] as! String)
        try require(try fileDigest(url) == value["sha256"] as? String, "Saved diagnostic payload changed: \(url.lastPathComponent)")
      }
      report["frozenSHA256After"] = pins.values.sorted { ($0["path"] as! String) < ($1["path"] as! String) }
      report["allPinsUnchanged"] = true
      report["allSavedPayloadHashesVerified"] = true
      await native.reset()
      report["processorResetAfterMatrix"] = true
      report["complete"] = true; report["passed"] = true; report["completedPhase"] = "complete"
      report.removeValue(forKey: "currentSourceFrameIndex"); report.removeValue(forKey: "currentReplayOrdinal")
      try publish()
    } catch {
      report["failure"] = error.localizedDescription
      report["complete"] = false; report["passed"] = false
      report["availablePins"] = pins.values.sorted { ($0["path"] as! String) < ($1["path"] as! String) }
      try? publish()
      if let processor { await processor.reset() }
      throw error
    }
  }
}
#endif
