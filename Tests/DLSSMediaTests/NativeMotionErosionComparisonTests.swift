import CryptoKit
import Foundation
import MLX
import XCTest
@testable import DLSSMLX

/// Opt-in isolated comparison; no new optical flow, decoding, model or playback.
final class NativeMotionErosionComparisonTests: XCTestCase, @unchecked Sendable {
  private let costSHA = "4bbca00854a99638a7a007a9ef54afe886dac49284c9354bb92f7ceb65bbdd5f"
  private let width = 1920, height = 1080

  private struct Pin: Decodable {
    let path: String
    let bytes: Int
    let sha256: String
  }
  private struct Retained: Decodable {
    let name: String
    let sourceFrameIndex: Int?
    let input: Pin
    let expected: Pin
  }
  private struct Cost: Decodable {
    let passed: Bool
    let completedPhase: String
    let allErosionOutputsExact: Bool
    let frozenFilesUnchanged: Bool
    let width: Int
    let height: Int
    let capturedMasks: [Retained]
    let syntheticControls: [Retained]
    let frozenSHA256Before: [String: String]
  }
  private struct Mask {
    let name: String
    let provenance: String
    let sourceIndex: Int?
    let input: Data
    let expected: Data
    let inputURL: URL
    let expectedURL: URL
    let inputSHA: String
    let expectedSHA: String
  }
  private struct Tiny {
    let name: String
    let width: Int
    let height: Int
    let bits: [UInt32]
  }
  private struct Result {
    let bytes: Data
    let started: UInt64
    let ended: UInt64
    let memory: [String: UInt64]
    var seconds: Double { Double(ended - started) / 1e9 }
  }
  private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "NativeMotionErosionComparison", code: 1,
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
  private func data(_ bits: [UInt32]) -> Data { bits.withUnsafeBytes { Data($0) } }
  private func bits(_ data: Data) -> [UInt32] {
    data.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: 4).map { bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self) }
    }
  }
  private func nonfinite(_ data: Data) -> Int {
    data.withUnsafeBytes { bytes in
      stride(from: 0, to: bytes.count, by: 4).reduce(0) { count, offset in
        let bit = bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        return count + (Float(bitPattern: bit).isFinite ? 0 : 1)
      }
    }
  }
  private func save(_ data: Data, _ name: String, output: URL) throws -> [String: Any] {
    try data.write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
    return ["path": name, "bytes": data.count, "sha256": digest(data)]
  }
  private func pinRecord(_ url: URL, bytes: Int, sha: String) -> [String: Any] {
    ["path": url.path, "bytes": bytes, "sha256": sha]
  }
  private func params(width: Int, height: Int) -> MLXArray {
    let value = MLXArray([UInt32(width), UInt32(height), 0, 0, 0, 0, 0, 0])
    eval(value)
    return value
  }
  private func apply(_ arm: String, input: MLXArray, params: MLXArray) -> MLXArray {
    arm == "baseline" ? FrozenErosionProbeVideoMotion.originalErosion(input, params: params)
      : SeparableMotionErosion.apply(input, params: params)
  }
  private func timed(_ arm: String, input: MLXArray, params: MLXArray) -> Result {
    let started = DispatchTime.now().uptimeNanoseconds
    let output = apply(arm, input: input, params: params)
    eval(output)
    let ended = DispatchTime.now().uptimeNanoseconds
    let memory = MLXRuntimeDiagnostics.memorySnapshot()
    return Result(bytes: output.asData(access: .copy).data, started: started, ended: ended,
      memory: ["active": memory.activeBytes, "cache": memory.cacheBytes, "peakActive": memory.peakActiveBytes])
  }
  private func mismatch(_ a: Data, _ b: Data) -> [String: Any] {
    let left = bits(a), right = bits(b)
    var differences = [[String: Any]](), count = 0, maxError = 0.0, sum = 0.0, finitePairs = 0
    for (index, pair) in zip(left, right).enumerated() {
      if pair.0 != pair.1 {
        count += 1
        if differences.count < 32 { differences.append(["index": index, "leftBits": pair.0, "rightBits": pair.1]) }
      }
      let x = Float(bitPattern: pair.0), y = Float(bitPattern: pair.1)
      if x.isFinite && y.isFinite {
        let error = abs(Double(x) - Double(y))
        maxError = max(maxError, error); sum += error; finitePairs += 1
      }
    }
    return ["leftBytes": a.count, "rightBytes": b.count, "differentCommonScalars": count,
      "firstDifferences": differences, "finitePairCount": finitePairs,
      "maximumFinitePairAbsoluteError": maxError,
      "meanFinitePairAbsoluteError": finitePairs == 0 ? NSNull() as Any : sum / Double(finitePairs)]
  }
  private func oracle(_ tiny: Tiny) -> Data {
    var output = [UInt32](repeating: 0, count: tiny.bits.count)
    for y in 0..<tiny.height {
      for x in 0..<tiny.width {
        var valid = true
        for dy in -3...3 {
          for dx in -3...3 {
            let xx = min(tiny.width - 1, max(0, x + dx)), yy = min(tiny.height - 1, max(0, y + dy))
            if !(Float(bitPattern: tiny.bits[yy * tiny.width + xx]) > 0) { valid = false }
          }
        }
        if valid { output[y * tiny.width + x] = tiny.bits[y * tiny.width + x] }
      }
    }
    return data(output)
  }
  private func tinyControls() -> [Tiny] {
    let positive: (Int) -> [UInt32] = { n in (0..<n).map { (Float(1 + $0 % 7) / 8).bitPattern } }
    var column = positive(9)
    column[0] = Float(1e-30).bitPattern; column[8] = Float(-0.25).bitPattern
    var corners = positive(30)
    corners[0] = 0; corners[29] = 0x80000000
    var special = positive(99)
    special[0] = 0x7fc12345; special[98] = 0xff800000
    var subnormal = positive(323)
    subnormal[0] = 0x80000000
    subnormal[9 * 17 + 8] = 0x00000001
    subnormal[15 * 17 + 13] = 0x7f800000
    return [
      Tiny(name: "positive-infinity-single", width: 1, height: 1, bits: [0x7f800000]),
      Tiny(name: "column-tiny-positive-negative-edge", width: 1, height: 9, bits: column),
      Tiny(name: "row-nonbinary-positive", width: 9, height: 1, bits: positive(9)),
      Tiny(name: "corner-positive-negative-zero", width: 5, height: 6, bits: corners),
      Tiny(name: "nan-negative-infinity-edges", width: 9, height: 11, bits: special),
      Tiny(name: "subnormal-and-infinity-centers", width: 17, height: 19, bits: subnormal),
    ]
  }
  private func distribution(_ seconds: [Double]) -> [String: Any] {
    guard !seconds.isEmpty else { return ["count": 0] }
    let sorted = seconds.sorted()
    let middle = sorted.count / 2
    let median = sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    return ["count": sorted.count, "meanSeconds": seconds.reduce(0, +) / Double(seconds.count),
      "medianSeconds": median, "p95Seconds": sorted[Int(ceil(Double(sorted.count) * 0.95)) - 1],
      "minimumSeconds": sorted[0], "maximumSeconds": sorted[sorted.count - 1]]
  }

  func testMatchedSeparableErosion() throws {
    let env = ProcessInfo.processInfo.environment
    guard let outputPath = env["MLXDLSS_EROSION_COMPARISON_OUTPUT"] else {
      throw XCTSkip("Opt-in separable erosion comparison")
    }
    #if DEBUG
    throw NSError(domain: "NativeMotionErosionComparison", code: 2,
      userInfo: [NSLocalizedDescriptionKey: "Comparison requires release configuration"])
    #endif
    let inputPath = try XCTUnwrap(env["MLXDLSS_EROSION_COMPARISON_INPUT_REPORT"])
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { root.deleteLastPathComponent() }
    let output = URL(fileURLWithPath: outputPath).standardizedFileURL
    let inputURL = URL(fileURLWithPath: inputPath).standardizedFileURL
    try require(!FileManager.default.fileExists(atPath: output.path) &&
      (try? output.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true, "Require new output directory")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    var report: [String: Any] = ["schemaVersion": 1, "passed": false,
      "actualBackend": "retained-mask-metal-replay", "modelLoaded": false, "newOpticalFlowEstimated": false,
      "completedPhase": "initializing", "expectedProgressUnits": 244, "width": width, "height": height,
      "erosionThreshold": "input>0.0f", "reliableFractionThreshold": "confidence>0.5",
      "intermediateLogicalBytes": width * height, "intermediateDType": "uint8",
      "timingScope": "Fresh arm apply through blocking eval; two candidate dispatches and intermediate allocation included; input/params materialization and readback/hash/file IO excluded",
      "limitations": ["Preflight executes both kernels; first-isolated is not cold, first use or compilation timing.",
        "Retained actual masks replay completed VT-derived confidence; no VT session or model runs here.",
        "CPU special-value disagreement is retained baseline behavior, not relaxed candidate parity or general IEEE equivalence.",
        "Isolated completion does not reproduce pipeline overlap or prove full-motion/NativeOpticalFlow benefit.",
        "No temporal-quality, Live, M5 or hard-memory acceptance. Readback/report writes affect conditions outside timers.",
        "Sampled MLX active/cache/peak are allocator observations; external watcher records RSS/operating conditions."]]
    var frozen = [String: String](), retainedRows = [[String: Any]]()
    var preflight = [[String: Any]](), samples = [[String: Any]](), comparisons = [[String: Any]]()
    var masks = [Mask](), evaluated = [MLXArray]()
    func publish() throws {
      var value = report
      value["frozenSHA256Before"] = frozen
      value["retainedMasks"] = retainedRows; value["preflight"] = preflight
      value["samples"] = samples; value["comparisons"] = comparisons
      value["progressUnits"] = preflight.count + samples.count
      value["preflightCount"] = preflight.count
      value["mainErosionCalls"] = samples.count; value["pairedComparisons"] = comparisons.count
      value["measuredCalls"] = samples.filter { $0["measured"] as? Bool == true }.count
      let measured = comparisons.filter { $0["measured"] as? Bool == true }
      value["measuredPairs"] = measured.count
      value["measuredNaturalPairs"] = measured.filter { $0["maskProvenance"] as? String == "natural" }.count
      value["measuredSyntheticPairs"] = measured.filter { $0["maskProvenance"] as? String == "synthetic" }.count
      try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        .write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    let priorCache = MLXRuntimeDiagnostics.cacheLimitBytes
    report["priorMLXCachePolicyBytes"] = priorCache
    defer {
      do {
        try MLXRuntimeDiagnostics.setCacheLimitBytes(priorCache)
        report["restoredMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
        if MLXRuntimeDiagnostics.cacheLimitBytes != priorCache {
          report["passed"] = false; XCTFail("Cache policy restoration differs")
        }
      } catch {
        report["passed"] = false; report["cacheRestorationError"] = error.localizedDescription
        XCTFail("Cache policy restoration failed: \(error)")
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
      let costBytes = try Data(contentsOf: inputURL)
      try require(digest(costBytes) == costSHA, "Pinned cost report differs")
      frozen[inputURL.path] = costSHA
      report["inputReport"] = pinRecord(inputURL, bytes: costBytes.count, sha: costSHA)
      let cost = try JSONDecoder().decode(Cost.self, from: costBytes)
      try require(cost.passed && cost.completedPhase == "complete" && cost.allErosionOutputsExact &&
        cost.frozenFilesUnchanged && cost.width == width && cost.height == height &&
        cost.capturedMasks.map(\.sourceFrameIndex) == Array(1497...1507).map(Optional.some) &&
        cost.syntheticControls.map(\.name) == ["all-positive", "sparse-invalid", "dense-invalid"],
        "Retained report acceptance, geometry or inventory differs")

      // Require unchanged measured production/test sources and root runtime. The new
      // release test executable is expected to differ from the historical cost build.
      for (path, expected) in cost.frozenSHA256Before where
        path.contains("/Sources/") || path.contains("/Tests/DLSSMediaTests/") ||
        path.hasPrefix(root.appendingPathComponent(".build/debug/").path + "/") {
        try require(fileDigest(URL(fileURLWithPath: path)) == expected, "Measured source/runtime pin differs: \(path)")
        frozen[path] = expected
      }
      let motionPath = root.appendingPathComponent("vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoMotion.swift")
      let frozenPath = root.appendingPathComponent("vendor/MLX-DLSS/Tests/DLSSMediaTests/FrozenErosionProbeVideoMotion.swift")
      let marker = try NSRegularExpression(pattern: #"(?ms)^[ \t]*// EROSION-PROBE-BEGIN\n.*?^[ \t]*// EROSION-PROBE-END\n"#)
      let text = try String(contentsOf: frozenPath, encoding: .utf8)
      var reversed = marker.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
      for (copy, original) in [
        ("FrozenErosionProbeVideoMotion", "MLXVideoMotion"),
        ("mlxdlss_erosion_probe_flow_import", "mlxdlss_native_flow_import"),
        ("mlxdlss_erosion_probe_motion_quality", "mlxdlss_native_motion_quality"),
        ("mlxdlss_erosion_probe_motion_erode", "mlxdlss_native_motion_erode"),
        ("mlxdlss_erosion_probe_guide_resize", "mlxdlss_native_guide_resize")] {
        reversed = reversed.replacingOccurrences(of: copy, with: original)
      }
      try require(Data(reversed.utf8) == Data(contentsOf: motionPath), "Frozen full-motion source reversal failed")
      report["reversedProductionMotionSHA256"] = digest(Data(reversed.utf8))
      let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL)
      report["actualTestExecutable"] = executable.path
      let bundleMetallib = executable.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
      report["actualTestBundleMetallib"] = bundleMetallib.path
      for url in [executable, bundleMetallib, root.appendingPathComponent("vendor/MLX-DLSS/.build/release/mlx.metallib"),
        root.appendingPathComponent("vendor/MLX-DLSS/Tests/DLSSMediaTests/SeparableMotionErosion.swift"),
        URL(fileURLWithPath: #filePath)] { frozen[url.path] = try fileDigest(url) }

      let base = inputURL.deletingLastPathComponent().resolvingSymlinksInPath()
      func read(_ entry: Pin) throws -> (URL, Data) {
        let url = base.appendingPathComponent(entry.path).resolvingSymlinksInPath()
        try require(!entry.path.hasPrefix("/") && url.path.hasPrefix(base.path + "/"), "Mask must be contained in cost output")
        let bytes = try Data(contentsOf: url)
        try require(entry.bytes == width * height * 4 && bytes.count == entry.bytes && digest(bytes) == entry.sha256,
          "Retained mask bytes/hash differ")
        frozen[url.path] = entry.sha256
        return (url, bytes)
      }
      for (kind, entries) in [("natural", cost.capturedMasks), ("synthetic", cost.syntheticControls)] {
        for entry in entries {
          let (aURL, a) = try read(entry.input), (bURL, b) = try read(entry.expected)
          try require(nonfinite(a) == 0 && nonfinite(b) == 0, "Unexpected nonfinite retained full mask")
          masks.append(Mask(name: entry.name, provenance: kind, sourceIndex: entry.sourceFrameIndex,
            input: a, expected: b, inputURL: aURL, expectedURL: bURL, inputSHA: entry.input.sha256, expectedSHA: entry.expected.sha256))
          retainedRows.append(["name": entry.name, "maskProvenance": kind,
            "sourceFrameIndex": entry.sourceFrameIndex as Any? ?? NSNull(),
            "input": pinRecord(aURL, bytes: a.count, sha: entry.input.sha256),
            "expected": pinRecord(bURL, bytes: b.count, sha: entry.expected.sha256)])
        }
      }
      try require(masks.count == 14 && Set(masks.map(\.name)).count == 14, "Expected14 distinct masks")
      try MLXRuntimeDiagnostics.setCacheLimitBytes(256 * 1024 * 1024)
      report["benchmarkMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
      report["completedPhase"] = "tiny-preflight"
      try publish()
      for tiny in tinyControls() {
        report["currentCase"] = tiny.name
        let inputBytes = data(tiny.bits), cpu = oracle(tiny)
        let input = MLXArray(inputBytes, [1, tiny.height, tiny.width, 1], dtype: .float32)
        let p = params(width: tiny.width, height: tiny.height)
        eval(input)
        try require(input.asData(access: .copy).data == inputBytes, "Tiny materialization changed input bits")
        let a = apply("baseline", input: input, params: p), b = apply("candidate", input: input, params: p)
        eval(a, b)
        let aBytes = a.asData(access: .copy).data, bBytes = b.asData(access: .copy).data
        let specialInput = tiny.bits.contains {
          let value = Float(bitPattern: $0)
          return !value.isFinite || (value != 0 && abs(value) < Float.leastNormalMagnitude)
        }
        var row: [String: Any] = ["name": tiny.name, "kind": "tiny", "width": tiny.width, "height": tiny.height,
          "inputBits": tiny.bits, "cpuBits": bits(cpu), "baselineBits": bits(aBytes), "candidateBits": bits(bBytes),
          "baselineCandidateByteEqual": aBytes == bBytes, "cpuBaselineByteEqual": cpu == aBytes,
          "cpuCandidateByteEqual": cpu == bBytes, "containsNonfiniteOrSubnormalInput": specialInput,
          "cpuClassification": cpu == aBytes ? "agrees" : specialInput
            ? "baseline-CPU-disagreement-on-special-input-retained; mechanism-not-established"
            : "unexpected-finite-normal-CPU-disagreement-failure",
          "input": try save(inputBytes, "tiny-\(tiny.name)-input.f32", output: output),
          "cpu": try save(cpu, "tiny-\(tiny.name)-cpu.f32", output: output),
          "baseline": try save(aBytes, "tiny-\(tiny.name)-baseline.f32", output: output),
          "candidate": try save(bBytes, "tiny-\(tiny.name)-candidate.f32", output: output)]
        if aBytes != bBytes { row["baselineCandidateDifference"] = mismatch(aBytes, bBytes) }
        if cpu != aBytes { row["cpuBaselineDifference"] = mismatch(cpu, aBytes) }
        preflight.append(row); try publish()
        try require(aBytes.count == inputBytes.count && aBytes == bBytes, "Tiny candidate differs from baseline; all bits/payloads retained")
        try require(cpu == aBytes || specialInput, "Finite normal tiny control differs from CPU oracle; all bits/payloads retained")
        try require(input.asData(access: .copy).data == inputBytes, "Tiny input mutated")
      }
      report["completedPhase"] = "full-preflight"
      let p = params(width: width, height: height)
      for mask in masks {
        report["currentCase"] = mask.name
        let input = MLXArray(mask.input, [1, height, width, 1], dtype: .float32)
        eval(input)
        try require(input.asData(access: .copy).data == mask.input, "Full input materialization changed bits")
        evaluated.append(input)
        let a = apply("baseline", input: input, params: p), b = apply("candidate", input: input, params: p)
        eval(a, b)
        let aBytes = a.asData(access: .copy).data, bBytes = b.asData(access: .copy).data
        var row: [String: Any] = ["name": mask.name, "kind": "full", "maskProvenance": mask.provenance,
          "baselineCandidateByteEqual": aBytes == bBytes, "baselineExpectedByteEqual": aBytes == mask.expected,
          "candidateExpectedByteEqual": bBytes == mask.expected, "baselineSHA256": digest(aBytes),
          "candidateSHA256": digest(bBytes), "inputSHA256": mask.inputSHA, "expectedSHA256": mask.expectedSHA]
        if aBytes != bBytes || aBytes != mask.expected {
          row["baselinePayload"] = try save(aBytes, "preflight-\(mask.name)-baseline.f32", output: output)
          row["candidatePayload"] = try save(bBytes, "preflight-\(mask.name)-candidate.f32", output: output)
          row["baselineCandidateDifference"] = mismatch(aBytes, bBytes)
          row["expectedBaselineDifference"] = mismatch(mask.expected, aBytes)
        }
        preflight.append(row); try publish()
        try require(aBytes == mask.expected && bBytes == mask.expected, "Full preflight differs; payloads retained")
      }
      report["completedPhase"] = "isolated-comparison"
      report["preflightFinishedBeforeTiming"] = true
      try publish()
      for pass in 0..<8 {
        let phase = pass == 0 ? "first-isolated" : pass < 4 ? "warmup" : "measured"
        let order = pass.isMultiple(of: 2) ? Array(masks.indices) : Array(masks.indices.reversed())
        for (orderIndex, index) in order.enumerated() {
          let mask = masks[index], pairIndex = comparisons.count
          // Stable index, not traversal position: each mask alternates first arm.
          let arms = (pass + index).isMultiple(of: 2) ? ["baseline", "candidate"] : ["candidate", "baseline"]
          var results = [String: Result]()
          report["currentCase"] = mask.name; report["currentPass"] = pass
          for (armIndex, arm) in arms.enumerated() {
            report["currentArm"] = arm
            try publish()
            let result = timed(arm, input: evaluated[index], params: p)
            results[arm] = result
            var row: [String: Any] = ["pairIndex": pairIndex, "pass": pass, "phase": phase, "measured": pass >= 4,
              "orderIndex": orderIndex, "maskIndex": index, "armIndex": armIndex, "arm": arm, "maskName": mask.name,
              "maskProvenance": mask.provenance, "sourceFrameIndex": mask.sourceIndex as Any? ?? NSNull(),
              "inputSHA256": mask.inputSHA, "expectedSHA256": mask.expectedSHA, "outputSHA256": digest(result.bytes),
              "outputBytes": result.bytes.count, "expectedByteEqual": result.bytes == mask.expected,
              "outputNonfiniteScalars": nonfinite(result.bytes),
              "startUptimeNanoseconds": result.started, "endUptimeNanoseconds": result.ended,
              "erosionCompletedWallSeconds": result.seconds, "mlxBytes": result.memory]
            if result.bytes != mask.expected {
              row["mismatchPayload"] = try save(result.bytes, "timed-\(pass)-\(mask.name)-\(arm).f32", output: output)
              row["expectedDifference"] = mismatch(mask.expected, result.bytes)
            }
            samples.append(row); try publish()
          }
          let a = try XCTUnwrap(results["baseline"]), b = try XCTUnwrap(results["candidate"])
          var pair: [String: Any] = ["pairIndex": pairIndex, "pass": pass, "phase": phase, "measured": pass >= 4,
            "orderIndex": orderIndex, "maskIndex": index, "maskName": mask.name, "maskProvenance": mask.provenance,
            "sourceFrameIndex": mask.sourceIndex as Any? ?? NSNull(), "firstArm": arms[0],
            "baselineCandidateByteEqual": a.bytes == b.bytes,
            "baselineExpectedByteEqual": a.bytes == mask.expected, "candidateExpectedByteEqual": b.bytes == mask.expected,
            "baselineSHA256": digest(a.bytes), "candidateSHA256": digest(b.bytes),
            "baselineSeconds": a.seconds, "candidateSeconds": b.seconds,
            "candidateMinusBaselineSeconds": b.seconds - a.seconds]
          if a.bytes != b.bytes { pair["baselineCandidateDifference"] = mismatch(a.bytes, b.bytes) }
          comparisons.append(pair); try publish()
          try require(a.bytes == mask.expected && b.bytes == mask.expected,
            "Timed candidate/reference differs; completed arm records and mismatch payloads retained")
        }
      }
      for (mask, input) in zip(masks, evaluated) {
        try require(input.asData(access: .copy).data == mask.input, "Materialized retained input changed")
      }
      var balance = [[String: Any]]()
      for mask in masks {
        let rows = comparisons.filter { $0["measured"] as? Bool == true && $0["maskName"] as? String == mask.name }
        let baselineFirst = rows.filter { $0["firstArm"] as? String == "baseline" }.count
        let candidateFirst = rows.filter { $0["firstArm"] as? String == "candidate" }.count
        balance.append(["maskName": mask.name, "baselineFirst": baselineFirst, "candidateFirst": candidateFirst])
        try require(baselineFirst == 2 && candidateFirst == 2, "Per-mask measured arm order is unbalanced")
      }
      report["measuredOrderBalance"] = balance
      var distributions = [[String: Any]](), rounds = [[String: Any]]()
      for provenance in ["natural", "synthetic"] {
        let rows = comparisons.filter { $0["measured"] as? Bool == true && $0["maskProvenance"] as? String == provenance }
        distributions.append(["maskProvenance": provenance,
          "baseline": distribution(rows.compactMap { $0["baselineSeconds"] as? Double }),
          "candidate": distribution(rows.compactMap { $0["candidateSeconds"] as? Double }),
          "pairedDifference": distribution(rows.compactMap { $0["candidateMinusBaselineSeconds"] as? Double })])
        for pass in 4..<8 {
          let round = rows.filter { $0["pass"] as? Int == pass }
          rounds.append(["maskProvenance": provenance, "pass": pass,
            "baseline": distribution(round.compactMap { $0["baselineSeconds"] as? Double }),
            "candidate": distribution(round.compactMap { $0["candidateSeconds"] as? Double }),
            "pairedDifference": distribution(round.compactMap { $0["candidateMinusBaselineSeconds"] as? Double })])
        }
      }
      report["measuredDistributions"] = distributions; report["measuredRounds"] = rounds
      report["cpuOracleDisagreements"] = preflight.filter { $0["kind"] as? String == "tiny" && $0["cpuBaselineByteEqual"] as? Bool == false }.count
      report["allBaselineCandidateOutputsExact"] = true
      report["allFullOutputsMatchRetainedExpected"] = true
      report["allMaterializedInputsUnchanged"] = true
      try require(preflight.count == 20 && samples.count == 224 && comparisons.count == 112 &&
        comparisons.filter { $0["measured"] as? Bool == true }.count == 56, "Incomplete comparison")
      report["completedPhase"] = "complete"; report["passed"] = true
      report.removeValue(forKey: "currentCase"); report.removeValue(forKey: "currentPass"); report.removeValue(forKey: "currentArm")
      try publish()
    } catch {
      report["failure"] = error.localizedDescription
      try? publish()
      throw error
    }
  }
}
