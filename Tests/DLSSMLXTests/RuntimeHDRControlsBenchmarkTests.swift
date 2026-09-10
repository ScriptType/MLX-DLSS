import CryptoKit
import Foundation
import MLX
import DLSSCore
import XCTest
@testable import DLSSMLX

/// Opt-in, model-free comparison with the frozen template-parameter codec.
/// Run this method alone in a fresh release test process. Readback is not timed.
final class RuntimeHDRControlsBenchmarkTests: XCTestCase {
  private let baselineSHA = "6eeacf6699f1f0978b121811970a0ad77e53378ac197c33fde17681510685389"
  private let renames = [
    ("MLXNeuralRenderingDisplayCodec", "FrozenRuntimeHDRControlsCodec"),
    ("neuralRenderingDisplayCodecMetalHeader", "frozenRuntimeHDRControlsMetalHeader"),
    ("mlxdlss_display_codec_encode", "mlxdlss_runtime_controls_baseline_encode"),
    ("mlxdlss_display_codec_resolve", "mlxdlss_runtime_controls_baseline_resolve"),
  ]
  // Both arms receive these same Float literals; no independently calculated coefficients.
  private let settings: [(Float, Float)] = [
    (0.05, 0.10), (0.10, 0.25), (0.15, 0.50), (0.20, 0.75),
    (0.25, 1.00), (0.30, 0.15), (0.35, 0.35), (0.40, 0.65),
    (0.45, 0.85), (0.50, 0.05), (0.55, 0.20), (0.60, 0.40),
    (0.65, 0.60), (0.70, 0.80), (0.75, 1.00), (0.80, 0.30),
    (0.85, 0.45), (0.90, 0.55), (0.95, 0.70), (1.00, 0.90),
    (0.08, 0.12), (0.18, 0.32), (0.28, 0.52), (0.38, 0.72),
    (0.48, 0.92), (0.58, 0.18), (0.68, 0.38), (0.78, 0.58),
    (0.88, 0.78), (0.98, 0.98), (1.00, 0.50), (1.00, 1.00),
  ]

  private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "RuntimeHDRControlsBenchmark", code: 1,
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
  private func data(_ values: [Float]) -> Data { values.withUnsafeBytes { Data($0) } }
  private func tensor(_ data: Data, width: Int, height: Int) throws -> HostTensor {
    try HostTensor(descriptor: TensorDescriptor(name: "color", shape: [1, height, width, 3],
      dataType: .float32, layout: .nhwc), bytes: data)
  }

  private struct Comparison {
    let exact: Bool
    let finite: Bool
    let maximumAbsolute: Double
    let report: [String: Any]
  }
  private func comparison(_ baseline: Data, _ candidate: Data, width: Int) throws -> Comparison {
    try require(baseline.count == candidate.count && baseline.count % 12 == 0, "Output geometry mismatch")
    var different = 0, badA = 0, badB = 0
    var minimumA = Float.infinity, maximumA = -Float.infinity
    var minimumB = Float.infinity, maximumB = -Float.infinity
    var sumA = 0.0, sumB = 0.0, sumDifference = 0.0
    var maximumAbsolute = 0.0, maximumRelative = 0.0
    var maximumULP: UInt64 = 0
    var absoluteIndex = 0, relativeIndex = 0, ulpIndex = 0
    func ordered(_ value: Float) -> UInt32 {
      let bits = value.bitPattern
      return bits & 0x8000_0000 == 0 ? bits | 0x8000_0000 : ~bits
    }
    baseline.withUnsafeBytes { aRaw in candidate.withUnsafeBytes { bRaw in
      let a = aRaw.bindMemory(to: Float.self), b = bRaw.bindMemory(to: Float.self)
      for i in a.indices {
        let av = a[i], bv = b[i]
        if av.bitPattern != bv.bitPattern { different += 1 }
        if av.isFinite { minimumA = min(minimumA, av); maximumA = max(maximumA, av); sumA += Double(av) }
        else { badA += 1 }
        if bv.isFinite { minimumB = min(minimumB, bv); maximumB = max(maximumB, bv); sumB += Double(bv) }
        else { badB += 1 }
        guard av.isFinite && bv.isFinite else { continue }
        let difference = abs(Double(av) - Double(bv))
        let relative = difference / max(max(abs(Double(av)), abs(Double(bv))), 1e-30)
        let ab = UInt64(ordered(av)), bb = UInt64(ordered(bv))
        let ulp = ab > bb ? ab - bb : bb - ab
        sumDifference += difference
        if difference > maximumAbsolute { maximumAbsolute = difference; absoluteIndex = i }
        if relative > maximumRelative { maximumRelative = relative; relativeIndex = i }
        if ulp > maximumULP { maximumULP = ulp; ulpIndex = i }
      }
    } }
    func location(_ index: Int) -> [String: Int] {
      ["scalarIndex": index, "x": (index / 3) % width, "y": index / (3 * width), "channel": index % 3]
    }
    func range(_ minimum: Float, _ maximum: Float, _ sum: Double, _ bad: Int) -> [String: Any] {
      ["minimum": minimum.isFinite ? Double(minimum) as Any : NSNull(),
       "maximum": maximum.isFinite ? Double(maximum) as Any : NSNull(),
       "meanFinite": sum / Double(max(1, baseline.count / 4 - bad)), "nonfiniteScalars": bad]
    }
    let exact = baseline == candidate
    return Comparison(exact: exact, finite: badA == 0 && badB == 0, maximumAbsolute: maximumAbsolute,
      report: ["byteEqual": exact, "bytesPerOutput": baseline.count, "differentScalarBits": different,
        "baselineSHA256": digest(baseline), "candidateSHA256": digest(candidate),
        "baselineRange": range(minimumA, maximumA, sumA, badA),
        "candidateRange": range(minimumB, maximumB, sumB, badB),
        "maximumAbsoluteDifference": maximumAbsolute, "maximumAbsoluteLocation": location(absoluteIndex),
        "meanAbsoluteDifference": sumDifference / Double(max(1, baseline.count / 4)),
        "maximumRelativeDifference": maximumRelative, "maximumRelativeLocation": location(relativeIndex),
        "relativeDenominator": "max(abs(baseline),abs(candidate),1e-30)",
        "maximumULPDistance": maximumULP, "maximumULPLocation": location(ulpIndex)])
  }

  private struct Fixture {
    let original: MLXVideoFrame
    let proxy: MLXVideoFrame
    let model: MLXVideoFrame
    let hashes: [String: String]
  }

  private func fullFixture(_ baseline: FrozenRuntimeHDRControlsCodec,
    _ candidate: MLXNeuralRenderingDisplayCodec, output: URL) throws -> (Fixture, [String: Any]) {
    let width = 1920, height = 1080
    let palette: [[Float]] = [[0, 0, 0], [0.001, 0.002, 0.004], [4, 2, 1],
      [203, 203, 203], [1000, 1000, 1000], [1000, 1, 0], [0, 800, 20],
      [-0.5, 20, 800], [35, 12, 7], [2000, 400, 100], [5, 60, 20], [0.1, 0.2, 0.3]]
    var original = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height { for x in 0..<width {
      let tile = palette[((x / 160) + (y / 90) * 3) % palette.count]
      let scale = Float(64 + (x * 13 + y * 7) % 193) / 128
      for c in 0..<3 { original[(y * width + x) * 3 + c] = tile[c] * scale }
    } }
    let originalBytes = data(original)
    let frame = try MLXVideoFrame(rgb: originalBytes, width: width, height: height)
    let config = NeuralRenderingDisplayCodecConfiguration(whitePoint: 203, maximumLuminanceRatio: 2,
      workingPrimaries: .bt2020)
    let proxy = baseline.encode(frame, configuration: config)
    let proxyBytes = proxy.copyRGBData()
    let encodedCheck = try comparison(proxyBytes, candidate.encode(frame, configuration: config).copyRGBData(), width: width)
    try require(encodedCheck.exact && encodedCheck.finite, "Unchanged encode path is not exact/finite")
    var model = [Float](repeating: 0, count: original.count)
    proxyBytes.withUnsafeBytes { raw in
      let values = raw.bindMemory(to: Float.self)
      for i in values.indices {
        let gain: Float = i % 3 == 0 ? 0.88 : i % 3 == 1 ? 0.94 : 0.90
        model[i] = min(1, max(0, values[i] * gain + (i % 3 == 1 ? 0.015 : 0.025)))
      }
    }
    let modelBytes = data(model)
    let inputBytes = ["original": originalBytes, "proxy": proxyBytes, "model": modelBytes]
    for (name, bytes) in inputBytes {
      try bytes.write(to: output.appendingPathComponent("fixture-\(name).rgb32f"), options: .withoutOverwriting)
    }
    let hashes = inputBytes.mapValues(digest)
    return (Fixture(original: frame, proxy: proxy,
      model: try MLXVideoFrame(rgb: modelBytes, width: width, height: height), hashes: hashes),
      ["width": width, "height": height, "layout": "RGB interleaved Float32 native little-endian",
       "sourceKind": "Deterministic synthetic HDR palette and spatial ramp; no model inference or natural-content quality claim",
       "originalUnits": "Absolute linear BT.2020 nits", "proxyAndModelUnits": "bounded sRGB BT.709",
       "inputSHA256": hashes, "bytesPerInput": originalBytes.count, "encodeComparison": encodedCheck.report])
  }

  func testMatchedNativeResolveRuntimeControls() throws {
    guard let outputPath = ProcessInfo.processInfo.environment["MLXDLSS_RUNTIME_HDR_CONTROLS_OUTPUT"] else {
      throw XCTSkip("Opt-in release runtime HDR controls benchmark")
    }
    #if DEBUG
    throw NSError(domain: "RuntimeHDRControlsBenchmark", code: 2,
      userInfo: [NSLocalizedDescriptionKey: "Benchmark requires release configuration"])
    #endif
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { root.deleteLastPathComponent() }
    let output = URL(fileURLWithPath: outputPath)
    try require(!FileManager.default.fileExists(atPath: output.path) &&
      (try? output.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true,
      "Benchmark output must be a new path, not an existing directory or symlink")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    var report: [String: Any] = ["schemaVersion": 1, "passed": false,
      "scope": "Completed native display resolve stage; baseline template versus immutable runtime effect parameters",
      "mainResolveCallsExpected": 192, "groups": 4, "settingsPerGroup": 8, "repeatsAfterFirstUse": 2,
      "modelLoaded": false, "exactnessRequired": true,
      "limitations": ["Settings-first-use is not guaranteed cold driver/disk cache; run method alone in a fresh process",
        "Candidate shares a program across settings; its first compile sample is retained",
        "Wall timing includes native resolve enqueue, output construction and evaluation, not CPU readback or hashing",
        "Readback, comparison, progress publication and synthetic input influence operating conditions outside timed intervals",
        "No inference, complete playback, Live, physical display or temporal-quality claim"]]
    var records: [[String: Any]] = [], pairs: [[String: Any]] = [], controls: [[String: Any]] = []
    var frozen: [String: String] = [:]
    let priorCache = MLXRuntimeDiagnostics.cacheLimitBytes
    report["priorMLXCachePolicyBytes"] = priorCache
    func publish() throws {
      var snapshot = report
      snapshot["samples"] = records; snapshot["comparisons"] = pairs; snapshot["numericalControls"] = controls
      try JSONSerialization.data(withJSONObject: snapshot, options: [.prettyPrinted, .sortedKeys])
        .write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    defer {
      do {
        try MLXRuntimeDiagnostics.setCacheLimitBytes(priorCache)
        report["restoredMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
      } catch { report["passed"] = false; report["cacheRestorationError"] = error.localizedDescription }
      var after: [String: String] = [:]
      for (path, expected) in frozen {
        do { after[path] = try fileDigest(URL(fileURLWithPath: path)) }
        catch { report["hashVerificationError"] = error.localizedDescription }
        if after[path] != expected { report["passed"] = false }
      }
      report["frozenSHA256After"] = after; report["frozenFilesUnchanged"] = !frozen.isEmpty && after == frozen
      try? publish()
    }
    do {
      let copyPath = "vendor/MLX-DLSS/Tests/DLSSMLXTests/RuntimeHDRControlsBaselineCodec.swift"
      var restored = try String(contentsOf: root.appendingPathComponent(copyPath), encoding: .utf8)
      try require(restored.hasPrefix("@testable import DLSSMLX\n"), "Unexpected baseline imports")
      restored.removeFirst("@testable import DLSSMLX\n".count)
      for (original, renamed) in renames { restored = restored.replacingOccurrences(of: renamed, with: original) }
      try require(digest(Data(restored.utf8)) == baselineSHA, "Baseline is not the exact archived source after reversing names/import")
      report["baselineReversedSHA256"] = baselineSHA
      report["baselineOnlyTransformations"] = renames.map { ["original": $0.0, "testName": $0.1] }
      let package = root.appendingPathComponent("vendor/MLX-DLSS")
      let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL, "Missing actual XCTest executable")
      let paths = [copyPath, "vendor/MLX-DLSS/Tests/DLSSMLXTests/RuntimeHDRControlsBenchmarkTests.swift",
        "vendor/MLX-DLSS/Sources/DLSSMLX/MLXNeuralRenderingDisplayCodec.swift",
        "vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoFrame.swift", "vendor/MLX-DLSS/Sources/DLSSMLX/MLXRuntimeDiagnostics.swift",
        "vendor/MLX-DLSS/Sources/DLSSCore/NeuralRenderingDisplayCodec.swift",
        "vendor/MLX-DLSS/Tests/DLSSMLXTests/MLXNeuralRenderingDisplayCodecTests.swift"]
      var pinURLs = paths.map { root.appendingPathComponent($0) }
      pinURLs += [executable, package.appendingPathComponent(".build/release/mlx.metallib")]
      for url in pinURLs { frozen[url.path] = try fileDigest(url) }
      report["frozenSHA256Before"] = frozen
      report["actualTestExecutable"] = executable.path
      try MLXRuntimeDiagnostics.setCacheLimitBytes(256 * 1024 * 1024)
      report["benchmarkMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
      let baseline = FrozenRuntimeHDRControlsCodec(), candidate = MLXNeuralRenderingDisplayCodec()
      try publish()
      let (fixture, fixtureReport) = try fullFixture(baseline, candidate, output: output)
      report["fixture"] = fixtureReport
      for name in ["original", "proxy", "model"] {
        frozen[output.appendingPathComponent("fixture-\(name).rgb32f").path] = fixture.hashes[name]
      }
      report["frozenSHA256Before"] = frozen
      var exact = true, savedMismatch = false, firstHashes: [String: String] = [:]
      var mismatchFiles: [String: String] = [:]
      for group in 0..<4 {
        for repetition in 0..<3 {
          for local in 0..<8 {
            let setting = group * 8 + local
            let (strength, colour) = settings[setting]
            let configuration = NeuralRenderingDisplayCodecConfiguration(whitePoint: 203,
              transferStrength: strength, colorStrength: colour, maximumLuminanceRatio: 2, workingPrimaries: .bt2020)
            let order = (setting + repetition).isMultiple(of: 2) ? ["baseline", "candidate"] : ["candidate", "baseline"]
            var bytes: [String: Data] = [:]
            for (orderIndex, arm) in order.enumerated() {
              let begin = DispatchTime.now().uptimeNanoseconds
              let frame = arm == "baseline"
                ? baseline.resolve(proxy: fixture.proxy, model: fixture.model, original: fixture.original, configuration: configuration)
                : candidate.resolve(proxy: fixture.proxy, model: fixture.model, original: fixture.original, configuration: configuration)
              // MLXVideoFrame.init completes eval; zero-strength separately returns the original.
              let elapsed = Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e9
              let memory = MLXRuntimeDiagnostics.memorySnapshot()
              let raw = frame.copyRGBData()
              let hash = digest(raw), key = "\(setting)-\(arm)"
              let stableRepeat = firstHashes[key].map { $0 == hash } ?? true
              if repetition == 0 { firstHashes[key] = hash }
              exact = exact && stableRepeat
              records.append(["group": group, "setting": setting, "repetition": repetition,
                "phase": repetition == 0 ? "settings-first-use" : "warmed-repeat", "arm": arm, "orderIndex": orderIndex,
                "transferStrength": strength, "colorStrength": colour,
                "transferStrengthBits": strength.bitPattern, "colorStrengthBits": colour.bitPattern,
                "nativeResolveCompletedWallSeconds": elapsed, "outputSHA256": hash,
                "repeatMatchesFirstUse": stableRepeat,
                "mlxBytes": ["active": memory.activeBytes, "cache": memory.cacheBytes, "peakActive": memory.peakActiveBytes]])
              bytes[arm] = raw
            }
            let match = try comparison(try XCTUnwrap(bytes["baseline"]), try XCTUnwrap(bytes["candidate"]), width: 1920)
            var row = match.report
            row["group"] = group; row["setting"] = setting; row["repetition"] = repetition
            exact = exact && match.exact
            if !match.exact || !match.finite {
              var retained: [String: String] = [:]
              for arm in order {
                let raw = try XCTUnwrap(bytes[arm]), hash = digest(raw)
                if mismatchFiles[hash] == nil {
                  let name = "mismatch-\(hash).rgb32f"
                  try raw.write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
                  mismatchFiles[hash] = name
                }
                retained[arm] = mismatchFiles[hash]
              }
              row["retainedRawOutputs"] = retained
              if !savedMismatch {
                report["firstMismatch"] = ["setting": setting, "repetition": repetition]
                savedMismatch = true
              }
            }
            pairs.append(row)
            report["allMainOutputsExact"] = exact
            try publish()
            try require(match.finite, "Nonfinite paired output; raw first mismatch retained")
          }
          report["allMainOutputsExact"] = exact
          try publish()
        }
      }
      // Small independent CPU-reference controls do not contaminate the first-use timing sequence.
      let originalBytes = data([0, 0, 0, 0.00001, 0.00002, 0.00003, -0.5, 20, 800,
        1000, 1, 0, 0, 800, 20, 203, 203, 203, 35, 12, 7, 2000, 400, 100])
      let proxyBytes = data([0, 0, 0, 0.00001, 0.00002, 0.00003, 0.04045, 0.1, 0.7,
        0.8, 0.01, 0.2, 0.02, 0.9, 0.1, 0.7, 0.7, 0.7, 0.2, 0.15, 0.1, 0.9, 0.7, 0.4])
      let modelBytes = data([0, 0, 0, 0.00002, 0.00001, 0.00004, 0.03, 0.2, 0.8,
        0.6, 0.03, 0.25, 0.04, 0.75, 0.15, 0.8, 0.7, 0.6, 0.18, 0.16, 0.12, 0.8, 0.75, 0.45])
      let smallProxy = try MLXVideoFrame(rgb: proxyBytes, width: 8, height: 1)
      let smallModel = try MLXVideoFrame(rgb: modelBytes, width: 8, height: 1)
      var cpuPassed = true, bypasses = 0
      for primaries in [NeuralRenderingDisplayCodecConfiguration.WorkingPrimaries.bt709, .bt2020] {
        // Generic BT.709 controls use the existing normalized white1.5 reference range.
        let inputBytes = primaries == .bt2020 ? originalBytes : data(originalBytes.withUnsafeBytes {
          $0.bindMemory(to: Float.self).map { $0 / 203 }
        })
        let smallOriginal = try MLXVideoFrame(rgb: inputBytes, width: 8, height: 1)
        for displayReferred in [false, true] { for strength in [Float(0), 1] { for colour in [Float(0), 1] {
          let configuration = NeuralRenderingDisplayCodecConfiguration(whitePoint: primaries == .bt2020 ? 203 : 1.5,
            transferStrength: strength, colorStrength: colour, maximumLuminanceRatio: 2,
            inputIsDisplayReferred: displayReferred, workingPrimaries: primaries)
          let old = baseline.resolve(proxy: smallProxy, model: smallModel, original: smallOriginal, configuration: configuration)
          let new = candidate.resolve(proxy: smallProxy, model: smallModel, original: smallOriginal, configuration: configuration)
          let oldBytes = old.copyRGBData(), newBytes = new.copyRGBData()
          let match = try comparison(oldBytes, newBytes, width: 8)
          let cpu = try NeuralRenderingDisplayCodec.resolve(proxy: tensor(proxyBytes, width: 8, height: 1),
            model: tensor(modelBytes, width: 8, height: 1), original: tensor(inputBytes, width: 8, height: 1), configuration: configuration)
          let cpuMatch = try comparison(cpu.bytes, newBytes, width: 8)
          let tolerance = primaries == .bt2020 ? 0.0005 : 0.00002
          let cpuOK = cpuMatch.finite && cpuMatch.maximumAbsolute <= tolerance
          let bypassOK = strength != 0 || (old === smallOriginal && new === smallOriginal && newBytes == inputBytes)
          if strength == 0 { bypasses += 1 }
          exact = exact && match.exact && bypassOK; cpuPassed = cpuPassed && cpuOK
          var row = match.report
          row["workingPrimaries"] = primaries == .bt2020 ? "bt2020" : "bt709"
          row["inputIsDisplayReferred"] = displayReferred; row["transferStrength"] = strength; row["colorStrength"] = colour
          row["zeroBypassObjectAndBytesExact"] = bypassOK
          row["cpuReference"] = cpuMatch.report; row["existingCPUAbsoluteTolerance"] = tolerance; row["cpuReferencePassed"] = cpuOK
          controls.append(row)
          if !match.exact || !cpuOK || !bypassOK {
            let index = controls.count - 1
            for (name, bytes) in [("baseline", oldBytes), ("candidate", newBytes), ("cpu", cpu.bytes)] {
              try bytes.write(to: output.appendingPathComponent("control-\(index)-\(name).rgb32f"), options: .withoutOverwriting)
            }
          }
          try require(match.finite && cpuMatch.finite, "Nonfinite small control output")
        } } }
      }
      for (name, frame) in [("original", fixture.original), ("proxy", fixture.proxy), ("model", fixture.model)] {
        try require(digest(frame.copyRGBData()) == fixture.hashes[name], "Retained fixture changed: \(name)")
      }
      report["inputFramesUnchanged"] = true; report["allOutputsExact"] = exact
      report["cpuReferenceControlsPassed"] = cpuPassed; report["zeroBypassControls"] = bypasses
      for (path, expected) in frozen {
        try require(fileDigest(URL(fileURLWithPath: path)) == expected, "Frozen file changed: \(path)")
      }
      report["passed"] = exact && cpuPassed && records.count == 192 && pairs.count == 96 && controls.count == 16 && bypasses == 8
      try publish()
      try require(report["passed"] as? Bool == true, "Runtime controls comparison failed; all timings, deltas and first mismatched raw pair retained")
    } catch {
      report["failure"] = error.localizedDescription
      try? publish()
      throw error
    }
  }
}
