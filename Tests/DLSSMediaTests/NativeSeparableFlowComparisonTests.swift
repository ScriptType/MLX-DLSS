import CryptoKit
import Foundation
import MLX
import XCTest
@testable import DLSSMedia
@testable import DLSSMLX

/// Matched complete prepare calls with two persistent, actual VT estimators.
final class NativeSeparableFlowComparisonTests: XCTestCase, @unchecked Sendable {
  private let inputSHA = "3c376d69b898a7fa0ee2474e233438c9071dbe3d63b3cece57d9b8e1bfa68181"
  private let originalFlowSHA = "9b705fc307cc46dec9d50453a0066e57970c806e92ff5111d0aba4f465fc590d"
  private struct Time: Decodable { let value: Int64; let timescale: Int32 }
  private struct Pin: Decodable { let path: String; let bytes: Int; let sha256: String }
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
    let source: Pin
    let captureManifest: Pin
  }
  private struct Snapshot: Equatable {
    let vectors: Data
    let confidence: Data
    let reliableBits: UInt32
    let errorBits: UInt32
    let reset: Bool
  }
  private struct Result {
    let snapshot: Snapshot?
    let started: UInt64
    let ended: UInt64
    let memory: [String: UInt64]
    var seconds: Double { Double(ended - started) / 1e9 }
  }
  private struct Signature: Equatable {
    let vectors: String?
    let confidence: String?
    let reliableBits: UInt32?
    let errorBits: UInt32?
    let reset: Bool?
    var json: [String: Any] {
      ["hasMotion": vectors != nil, "vectorsSHA256": vectors as Any? ?? NSNull(),
        "confidenceSHA256": confidence as Any? ?? NSNull(), "reliableFractionBits": reliableBits as Any? ?? NSNull(),
        "warpedLumaErrorBits": errorBits as Any? ?? NSNull(), "reset": reset as Any? ?? NSNull()]
    }
  }
  private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "NativeSeparableFlowComparison", code: 1,
      userInfo: [NSLocalizedDescriptionKey: message]) }
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
  private func finite(_ bytes: Data) -> Bool {
    bytes.withUnsafeBytes { raw in
      stride(from: 0, to: raw.count, by: 4).allSatisfy {
        Float(bitPattern: raw.loadUnaligned(fromByteOffset: $0, as: UInt32.self)).isFinite
      }
    }
  }
  private func valid(_ value: Snapshot?) -> Bool {
    guard let value else { return true }
    return value.vectors.count == 1920 * 1080 * 8 && value.confidence.count == 1920 * 1080 * 4 &&
      finite(value.vectors) && finite(value.confidence) &&
      Float(bitPattern: value.reliableBits).isFinite && Float(bitPattern: value.errorBits).isFinite
  }
  private func signature(_ value: Snapshot?) -> Signature {
    Signature(vectors: value.map { digest($0.vectors) }, confidence: value.map { digest($0.confidence) },
      reliableBits: value?.reliableBits, errorBits: value?.errorBits, reset: value?.reset)
  }
  private func save(_ value: Snapshot, prefix: String, output: URL) throws -> [String: Any] {
    var files = [String: Any]()
    for (kind, bytes) in [("vectors", value.vectors), ("confidence", value.confidence)] {
      let name = "\(prefix)-\(kind).f32"
      try bytes.write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
      files[kind] = ["path": name, "bytes": bytes.count, "sha256": digest(bytes)]
    }
    files["reliableFractionBits"] = value.reliableBits
    files["warpedLumaErrorBits"] = value.errorBits; files["reset"] = value.reset
    return files
  }
  private func snapshot(vectors: MLXArray, confidence: MLXArray, reliable: Float, error: Float, reset: Bool) -> Snapshot {
    Snapshot(vectors: vectors.asData(access: .copy).data, confidence: confidence.asData(access: .copy).data,
      reliableBits: reliable.bitPattern, errorBits: error.bitPattern, reset: reset)
  }
  private func run(_ arm: String, baseline: NativeOpticalFlow, candidate: SeparableErosionNativeOpticalFlow,
    frame: MLXVideoFrame, index: Int) async throws -> Result {
    if arm == "baseline" {
      let started = DispatchTime.now().uptimeNanoseconds
      let motion = try await baseline.prepare(frame, index: index, sceneCutThreshold: 0.3)
      let ended = DispatchTime.now().uptimeNanoseconds
      let memory = MLXRuntimeDiagnostics.memorySnapshot()
      let value = motion.map { snapshot(vectors: $0.vectors, confidence: $0.confidence,
        reliable: $0.reliableFraction, error: $0.warpedLumaError, reset: $0.reset) }
      return Result(snapshot: value, started: started, ended: ended,
        memory: ["active": memory.activeBytes, "cache": memory.cacheBytes, "peakActive": memory.peakActiveBytes])
    }
    let started = DispatchTime.now().uptimeNanoseconds
    let motion = try await candidate.prepare(frame, index: index, sceneCutThreshold: 0.3)
    let ended = DispatchTime.now().uptimeNanoseconds
    let memory = MLXRuntimeDiagnostics.memorySnapshot()
    let value = motion.map { snapshot(vectors: $0.vectors, confidence: $0.confidence,
      reliable: $0.reliableFraction, error: $0.warpedLumaError, reset: $0.reset) }
    return Result(snapshot: value, started: started, ended: ended,
      memory: ["active": memory.activeBytes, "cache": memory.cacheBytes, "peakActive": memory.peakActiveBytes])
  }
  private func distribution(_ values: [Double]) -> [String: Any] {
    guard !values.isEmpty else { return ["count": 0] }
    let sorted = values.sorted(), middle = values.count / 2
    return ["count": values.count, "meanSeconds": values.reduce(0, +) / Double(values.count),
      "medianSeconds": values.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle],
      "p95Seconds": sorted[Int(ceil(Double(values.count) * 0.95)) - 1],
      "minimumSeconds": sorted[0], "maximumSeconds": sorted[sorted.count - 1]]
  }

  func testMatchedNativePrepare() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let outputPath = env["MLXDLSS_SEPARABLE_FLOW_OUTPUT"] else {
      throw XCTSkip("Opt-in complete NativeOpticalFlow.prepare comparison")
    }
    #if DEBUG
    throw NSError(domain: "NativeSeparableFlowComparison", code: 2,
      userInfo: [NSLocalizedDescriptionKey: "Comparison requires release configuration"])
    #endif
    let inputURL = URL(fileURLWithPath: try XCTUnwrap(env["MLXDLSS_SEPARABLE_FLOW_INPUTS"])).standardizedFileURL
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { root.deleteLastPathComponent() }
    let output = URL(fileURLWithPath: outputPath).standardizedFileURL
    try require(!FileManager.default.fileExists(atPath: output.path) &&
      (try? output.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true, "Require new output directory")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    var report: [String: Any] = ["schemaVersion": 1, "passed": false, "modelLoaded": false,
      "newOpticalFlowEstimated": true, "actualBackend": "videotoolbox", "completedPhase": "initializing",
      "expectedProgressUnits": 288, "width": 1920, "height": 1080,
      "workingWidth": 960, "workingHeight": 540, "flowWidth": 240, "flowHeight": 135,
      "pairedCases": 4, "traversalsPerCasePerArm": 3, "framesPerTraversal": 12,
      "warmupCallsPerTraversal": 3, "sceneCutThresholdBits": Float(0.3).bitPattern,
      "sessionObjectsPerArm": 1, "maximumRetainedDirectControlSnapshots": 6,
      "timingScope": "Outer prepare await through completed return; includes packing, CI resize, fresh VT flow and full motion assessment. No internal clocks added.",
      "randomAccessScope": "Expected submission mode inferred from reversed source plus prior public reset/index; private VT parameters are not instrumented.",
      "limitations": ["Two persistent sessions survive all traversals/cases; only the initial session call of each arm returns nil.",
        "Index<3 is excluded from warmed summaries in every traversal, matching prior pool schedule; it is not a driver-cold definition.",
        "Fresh VT estimates run independently in each arm. Full-array hashes/scalars/reset are checked for all144 pairs;24 selected pairs also compare complete bytes.",
        "At most six first-arm snapshots are retained per case; a mismatch elsewhere retains available second-arm payload and first-arm signature, not invented first-arm bytes.",
        "Untimed readback/hash/report work and distinct test/production modules/kernel names influence operating conditions.",
        "Captured reset/public nil/index-wrap checks do not directly observe private randomAccess or prove general cancellation/error behavior.",
        "No model/full-player/temporal-quality/Live/M5 or hard-memory acceptance is established."]]
    var frozen = [String: String](), sourceRows = [[String: Any]](), records = [[String: Any]]()
    var comparisons = [[String: Any]](), controls = [[String: Any]]()
    func publish() throws {
      var value = report
      value["frozenSHA256Before"] = frozen; value["sourceFrames"] = sourceRows
      value["samples"] = records; value["comparisons"] = comparisons; value["directControls"] = controls
      value["progressUnits"] = records.count; value["mainPrepareCalls"] = records.count
      value["comparedCalls"] = comparisons.count; value["directByteChecks"] = controls.count
      value["warmedCalls"] = records.filter { $0["measured"] as? Bool == true }.count
      value["warmedPairs"] = comparisons.filter { $0["measured"] as? Bool == true }.count
      value["nilCalls"] = records.filter { $0["hasMotion"] as? Bool == false }.count
      value["nilPairs"] = comparisons.filter { $0["hasMotion"] as? Bool == false }.count
      value["wrapCalls"] = records.filter { $0["wrapDiscontinuity"] as? Bool == true }.count
      try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        .write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    let priorCache = MLXRuntimeDiagnostics.cacheLimitBytes
    report["priorMLXCachePolicyBytes"] = priorCache
    defer {
      do {
        try MLXRuntimeDiagnostics.setCacheLimitBytes(priorCache)
        report["restoredMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
        if MLXRuntimeDiagnostics.cacheLimitBytes != priorCache { report["passed"] = false; XCTFail("Cache restoration differs") }
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
      report["frozenSHA256After"] = after; report["frozenFilesUnchanged"] = !frozen.isEmpty && after == frozen
      do { try publish() } catch { XCTFail("Final report write failed: \(error)") }
    }
    do {
      let bytes = try Data(contentsOf: inputURL)
      try require(digest(bytes) == inputSHA, "Pinned12-source manifest differs")
      frozen[inputURL.path] = inputSHA
      report["inputManifest"] = ["path": inputURL.path, "bytes": bytes.count, "sha256": inputSHA]
      let input = try JSONDecoder().decode(Inputs.self, from: bytes)
      try require(input.width == 1920 && input.height == 1080 &&
        input.frames.map(\.sourceFrameIndex) == Array(1496...1507), "Source geometry/inventory differs")
      let sourceURL = root.appendingPathComponent("vendor/MLX-DLSS/Sources/DLSSMedia/NativeOpticalFlow.swift")
      let copyURL = root.appendingPathComponent("vendor/MLX-DLSS/Tests/DLSSMediaTests/SeparableErosionNativeOpticalFlow.swift")
      let original = try Data(contentsOf: sourceURL)
      var copy = try String(contentsOf: copyURL, encoding: .utf8)
      let addition = "// SEPARABLE-FLOW-IMPORT-BEGIN\n@testable import DLSSMedia\n// SEPARABLE-FLOW-IMPORT-END\n"
      try require(copy.components(separatedBy: addition).count == 2, "Candidate import marker differs")
      copy = copy.replacingOccurrences(of: addition, with: "")
      for (candidate, production) in [
        ("SeparableErosionNativeOpticalFlow", "NativeOpticalFlow"),
        ("SeparableErosionVideoToolboxFlowSession", "VideoToolboxFlowSession"),
        ("SeparableErosionVideoMotion", "MLXVideoMotion")] {
        copy = copy.replacingOccurrences(of: candidate, with: production)
      }
      try require(digest(original) == originalFlowSHA && Data(copy.utf8) == original, "Entire flow-source reversal differs")
      report["reversedProductionFlowSHA256"] = digest(Data(copy.utf8))
      let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL)
      let bundleMetallib = executable.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
      report["actualTestExecutable"] = executable.path; report["actualTestBundleMetallib"] = bundleMetallib.path
      let relativePaths = [
        "Package.swift", "Package.resolved", "vendor/MLX-DLSS/Package.swift", "vendor/MLX-DLSS/Package.resolved",
        "vendor/MLX-DLSS/Sources/DLSSMedia/NativeOpticalFlow.swift", "vendor/MLX-DLSS/Sources/DLSSMedia/NativeImageIO.swift",
        "vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoFrame.swift", "vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoMotion.swift",
        "vendor/MLX-DLSS/Sources/DLSSMLX/MLXRuntimeDiagnostics.swift",
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/ErosionProbeNativeOpticalFlow.swift",
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/FrozenErosionProbeVideoMotion.swift",
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/NativeMotionErosionCostTests.swift",
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/SeparableMotionErosion.swift",
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/NativeMotionErosionComparisonTests.swift",
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/SeparableErosionVideoMotion.swift",
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/NativeSeparableMotionComparisonTests.swift",
        "vendor/MLX-DLSS/.build/release/mlx.metallib", ".build/debug/hdr-benchmark",
        ".build/debug/libFrameEngineShared.dylib", ".build/debug/mlx.metallib"]
      for url in relativePaths.map({ root.appendingPathComponent($0) }) +
        [copyURL, URL(fileURLWithPath: #filePath), executable, bundleMetallib] {
        frozen[url.path] = try fileDigest(url)
      }
      let requiredMotionSources = [
        "vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoMotion.swift": "d8878517b8e68ce8971c3c8e1813c92796303937ebde387cd875545b7b4e313e",
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/SeparableErosionVideoMotion.swift": "bd542090d8ccdef1f1b0dd03ce79d3e6a398482f397107f96b6d0be1a80ed257",
        "vendor/MLX-DLSS/Tests/DLSSMediaTests/SeparableMotionErosion.swift": "217836ba32d79d63188d131f2c129e981b5506026f28706dcd796d462889d768"]
      for (path, expected) in requiredMotionSources {
        try require(frozen[root.appendingPathComponent(path).path] == expected, "Previously reviewed motion/helper source changed: \(path)")
      }
      report["requiredMotionSourceSHA256"] = requiredMotionSources
      for pin in [input.source, input.captureManifest] {
        let url = pin.path.hasPrefix("/") ? URL(fileURLWithPath: pin.path) : root.appendingPathComponent(pin.path)
        try require(fileDigest(url) == pin.sha256 && url.resourceValues(forKeys: [.fileSizeKey]).fileSize == pin.bytes,
          "Original source/capture pin differs")
        frozen[url.path] = pin.sha256
      }
      try MLXRuntimeDiagnostics.setCacheLimitBytes(256 * 1024 * 1024)
      report["benchmarkMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
      report["completedPhase"] = "materialize-inputs"; try publish()
      var frames = [MLXVideoFrame]()
      for source in input.frames {
        let url = source.path.hasPrefix("/") ? URL(fileURLWithPath: source.path) : root.appendingPathComponent(source.path)
        let data = try Data(contentsOf: url)
        try require(data.count == source.bytes && data.count == 1920 * 1080 * 12 && digest(data) == source.sha256 && finite(data),
          "Input proxy bytes/hash/finite domain differs")
        try require(source.pts.timescale == 24000 && source.duration.timescale == 24000 && source.duration.value == 1001,
          "Original rational source timing differs")
        frozen[url.path] = source.sha256
        let frame = try MLXVideoFrame(rgb: data, width: input.width, height: input.height)
        try require(frame.copyRGBData() == data, "Materialized source changed bytes")
        frames.append(frame)
        sourceRows.append(["sourceFrameIndex": source.sourceFrameIndex,
          "pts": ["value": source.pts.value, "timescale": Int64(source.pts.timescale)],
          "duration": ["value": source.duration.value, "timescale": Int64(source.duration.timescale)],
          "payload": ["path": url.path, "bytes": data.count, "sha256": source.sha256],
          "materializedByteEqualBefore": true])
        try publish()
      }
      let baseline = try NativeOpticalFlow(width: input.width, height: input.height, mode: .videoToolbox)
      let candidate = try SeparableErosionNativeOpticalFlow(width: input.width, height: input.height, mode: .videoToolbox)
      try require(baseline.backend == "videotoolbox" && candidate.backend == "videotoolbox", "Both actual backends must be VT")
      report["actualBackends"] = ["baseline": baseline.backend, "candidate": candidate.backend]
      var lastIndex = [String: Int](), lastReset = [String: Bool]()
      report["completedPhase"] = "matched-prepare"; try publish()
      for pairedCase in 0..<4 {
        let order = pairedCase.isMultiple(of: 2) ? ["baseline", "candidate"] : ["candidate", "baseline"]
        var references = [Signature](), referenceSeconds = [Double]()
        var directReferences = [Int: Snapshot]()
        for (caseOrder, arm) in order.enumerated() {
          for traversal in 0..<3 {
            for index in frames.indices {
              let source = input.frames[index], ordinal = traversal * frames.count + index
              let hasPrevious = lastIndex[arm] != nil
              let wrapped = lastIndex[arm].map { $0 + 1 != source.sourceFrameIndex } ?? false
              let expectedRandom: Bool? = hasPrevious ? ((lastReset[arm] ?? true) || wrapped) : nil
              report["currentCase"] = pairedCase; report["currentArm"] = arm
              report["currentTraversal"] = traversal; report["currentSourceFrameIndex"] = source.sourceFrameIndex
              try publish()
              let result = try await run(arm, baseline: baseline, candidate: candidate, frame: frames[index], index: source.sourceFrameIndex)
              let snap = result.snapshot, sig = signature(snap)
              let outputValid = valid(snap), nilCorrect = (snap != nil) == hasPrevious
              var sample: [String: Any] = ["pairedCase": pairedCase, "caseOrder": caseOrder, "arm": arm,
                "traversal": traversal, "sourceFrameIndex": source.sourceFrameIndex, "ordinalWithinCaseArm": ordinal,
                "pts": ["value": source.pts.value, "timescale": Int64(source.pts.timescale)],
                "duration": ["value": source.duration.value, "timescale": Int64(source.duration.timescale)],
                "inputSHA256": source.sha256, "previousSourceFrameIndex": lastIndex[arm] as Any? ?? NSNull(),
                "wrapDiscontinuity": wrapped, "expectedRandomAccessFromPublicHistory": expectedRandom as Any? ?? NSNull(),
                "hasMotion": snap != nil, "nilHistoryCorrect": nilCorrect, "outputFiniteAndExtentValid": outputValid,
                "phase": index < 3 ? "traversal-warmup" : "measured", "measured": index >= 3, "excludedWarmup": index < 3,
                "maskProvenance": "natural", "outerStartUptimeNanoseconds": result.started, "outerEndUptimeNanoseconds": result.ended,
                "prepareCompletedWallSeconds": result.seconds, "mlxBytes": result.memory, "signature": sig.json]
              if let snap {
                sample["vectorsBytes"] = snap.vectors.count; sample["confidenceBytes"] = snap.confidence.count
                sample["reset"] = snap.reset; lastReset[arm] = snap.reset
              }
              lastIndex[arm] = source.sourceFrameIndex
              var equal = true, firstDirect: Snapshot?
              if caseOrder == 0 {
                references.append(sig); referenceSeconds.append(result.seconds)
                if index == 2 || index == 3, let snap { directReferences[ordinal] = snap }
              } else {
                let first = references[ordinal]
                equal = first == sig
                var row: [String: Any] = ["pairedCase": pairedCase, "traversal": traversal,
                  "sourceFrameIndex": source.sourceFrameIndex, "ordinalWithinCaseArm": ordinal,
                  "firstArm": order[0], "secondArm": arm, "firstSignature": first.json, "secondSignature": sig.json,
                  "matchedExact": equal, "hasMotion": snap != nil, "measured": index >= 3,
                  "phase": index < 3 ? "traversal-warmup" : "measured", "maskProvenance": "natural",
                  "baselineSeconds": arm == "baseline" ? result.seconds : referenceSeconds[ordinal],
                  "candidateSeconds": arm == "candidate" ? result.seconds : referenceSeconds[ordinal],
                  "candidateMinusBaselineSeconds": arm == "candidate"
                    ? result.seconds - referenceSeconds[ordinal] : referenceSeconds[ordinal] - result.seconds]
                if index == 2 || index == 3 {
                  firstDirect = directReferences.removeValue(forKey: ordinal)
                  let direct = firstDirect != nil && snap != nil && firstDirect == snap
                  equal = equal && direct
                  row["directBytesEqual"] = direct; sample["directBytesEqual"] = direct
                  controls.append(["pairedCase": pairedCase, "traversal": traversal, "sourceFrameIndex": source.sourceFrameIndex,
                    "firstArm": order[0], "secondArm": arm, "directBytesEqual": direct,
                    "firstSignature": first.json, "secondSignature": sig.json])
                }
                comparisons.append(row); sample["matchedExact"] = equal
              }
              if !equal || !outputValid || !nilCorrect {
                if let snap { sample["availableOutputPayloads"] = try save(snap, prefix: "failure-\(records.count)-\(arm)", output: output) }
                if let firstDirect {
                  sample["firstArmPayloads"] = try save(firstDirect, prefix: "failure-\(records.count)-first-arm", output: output)
                } else if caseOrder == 1 {
                  sample["firstArmPayloadScope"] = "Only signature retained for this non-control or nil row; no unavailable bytes invented"
                }
              }
              records.append(sample)
              try publish()
              try require(equal && outputValid && nilCorrect, "Prepare output/signature/ownership mismatch; available payloads and all signatures retained")
            }
          }
        }
        try require(directReferences.isEmpty, "Direct reference snapshots were not released after matched comparisons")
      }
      report["completedPhase"] = "verify-inputs"
      var afterInputs = [[String: Any]]()
      for (source, frame) in zip(input.frames, frames) {
        let sha = digest(frame.copyRGBData())
        afterInputs.append(["sourceFrameIndex": source.sourceFrameIndex, "sha256After": sha, "expectedSHA256": source.sha256])
        try require(sha == source.sha256, "Retained input frame mutated")
      }
      report["materializedFramesAfter"] = afterInputs
      var cases = [[String: Any]](), traversals = [[String: Any]]()
      for pairedCase in 0..<4 {
        let rows = comparisons.filter { $0["pairedCase"] as? Int == pairedCase && $0["measured"] as? Bool == true }
        cases.append(["pairedCase": pairedCase, "firstArm": pairedCase.isMultiple(of: 2) ? "baseline" : "candidate",
          "baseline": distribution(rows.compactMap { $0["baselineSeconds"] as? Double }),
          "candidate": distribution(rows.compactMap { $0["candidateSeconds"] as? Double }),
          "pairedDifference": distribution(rows.compactMap { $0["candidateMinusBaselineSeconds"] as? Double })])
        for traversal in 0..<3 {
          let subset = rows.filter { $0["traversal"] as? Int == traversal }
          traversals.append(["pairedCase": pairedCase, "traversal": traversal,
            "baseline": distribution(subset.compactMap { $0["baselineSeconds"] as? Double }),
            "candidate": distribution(subset.compactMap { $0["candidateSeconds"] as? Double }),
            "pairedDifference": distribution(subset.compactMap { $0["candidateMinusBaselineSeconds"] as? Double })])
        }
      }
      report["warmedCaseDistributions"] = cases; report["warmedTraversalDistributions"] = traversals
      report["allCallDistributions"] = [
        "baseline": distribution(comparisons.compactMap { $0["baselineSeconds"] as? Double }),
        "candidate": distribution(comparisons.compactMap { $0["candidateSeconds"] as? Double }),
        "pairedDifference": distribution(comparisons.compactMap { $0["candidateMinusBaselineSeconds"] as? Double })]
      let warmed = comparisons.filter { $0["measured"] as? Bool == true }
      report["warmedDistributions"] = [
        "baseline": distribution(warmed.compactMap { $0["baselineSeconds"] as? Double }),
        "candidate": distribution(warmed.compactMap { $0["candidateSeconds"] as? Double }),
        "pairedDifference": distribution(warmed.compactMap { $0["candidateMinusBaselineSeconds"] as? Double })]
      let nilCalls = records.filter { $0["hasMotion"] as? Bool == false }.count
      let wraps = records.filter { $0["wrapDiscontinuity"] as? Bool == true }.count
      var resetCounts = [String: Int]()
      for row in records where row["reset"] as? Bool == true {
        let index = try XCTUnwrap(row["sourceFrameIndex"] as? Int)
        resetCounts[String(index), default: 0] += 1
      }
      report["observedResetCountsBySourceIndex"] = resetCounts
      report["expectedFixtureResetCountsBySourceIndex"] = ["1496": 22, "1498": 24]
      report["fixtureResetInventoryMatchesPriorObservation"] = resetCounts == ["1496": 22, "1498": 24]
      try require(records.count == 288 && comparisons.count == 144 && controls.count == 24 &&
        records.filter { $0["measured"] as? Bool == true }.count == 216 && warmed.count == 108 &&
        nilCalls == 2 && comparisons.filter { $0["hasMotion"] as? Bool == false }.count == 1 && wraps == 22,
        "Incomplete calls/controls or unexpected cold/wrap inventory")
      try require(resetCounts == ["1496": 22, "1498": 24], "Fixture reset inventory changed; retain and review actual baseline/candidate evidence")
      report["allComparedOutputsExact"] = true; report["allDirectControlsExact"] = true
      report["allOutputsFiniteAndExtentValid"] = true; report["allMaterializedInputsUnchanged"] = true
      report["passed"] = true; report["completedPhase"] = "complete"
      for key in ["currentCase", "currentArm", "currentTraversal", "currentSourceFrameIndex"] { report.removeValue(forKey: key) }
      try publish()
    } catch {
      report["failure"] = error.localizedDescription
      try? publish()
      throw error
    }
  }
}
