import CoreVideo
import CryptoKit
import Foundation
import MLX
import VideoToolbox
import XCTest
@testable import DLSSMedia
@testable import DLSSMLX

private struct AttributedFlowMotionSnapshot: Equatable {
  let vectors: Data
  let confidence: Data
  let reset: Bool
  let reliableFraction: Float
  let warpedLumaError: Float
  init(_ motion: MLXVideoMotion) {
    vectors = motion.vectors.asData(access: .copy).data
    confidence = motion.confidence.asData(access: .copy).data
    reset = motion.reset; reliableFraction = motion.reliableFraction; warpedLumaError = motion.warpedLumaError
  }
  var finite: Bool {
    [vectors, confidence].allSatisfy { $0.withUnsafeBytes { $0.bindMemory(to: Float.self).allSatisfy(\.isFinite) } }
      && reliableFraction.isFinite && warpedLumaError.isFinite
  }
  static func == (a: Self, b: Self) -> Bool {
    a.vectors == b.vectors && a.confidence == b.confidence && a.reset == b.reset &&
      a.reliableFraction.bitPattern == b.reliableFraction.bitPattern && a.warpedLumaError.bitPattern == b.warpedLumaError.bitPattern
  }
}

/// Opt in with MLXDLSS_FLOW_STAGE_INPUTS and MLXDLSS_FLOW_STAGE_ATTRIBUTION_OUTPUT.
/// Source attribution only: no alternate allocator, model or runtime algorithm.
final class NativeFlowStageAttributionTests: XCTestCase, @unchecked Sendable {
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
  private let originalFlowSHA = "9b705fc307cc46dec9d50453a0066e57970c806e92ff5111d0aba4f465fc590d"
  private func require(_ value: Bool, _ message: String) throws {
    if !value { throw NSError(domain: "NativeFlowStageAttribution", code: 1,
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
  private func arraySummary(_ data: Data) -> [String: Any] {
    var minimum = Float.infinity, maximum = -Float.infinity, sum = 0.0, nonfinite = 0
    data.withUnsafeBytes { raw in
      for value in raw.bindMemory(to: Float.self) {
        if value.isFinite { minimum = min(minimum, value); maximum = max(maximum, value); sum += Double(value) }
        else { nonfinite += 1 }
      }
    }
    return ["sha256": digest(data), "bytes": data.count, "nonfiniteComponents": nonfinite,
      "minimum": minimum.isFinite ? Double(minimum) as Any : NSNull(),
      "maximum": maximum.isFinite ? Double(maximum) as Any : NSNull(),
      "meanFinite": sum / Double(max(1, data.count / 4 - nonfinite))]
  }
  private func workingBytes(_ owner: MLXPixelBuffer) throws -> Data {
    let buffer = owner.buffer
    try require(!CVPixelBufferIsPlanar(buffer) && CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_64RGBAHalf,
      "Working buffer must be nonplanar RGBAHalf")
    try require(owner.width == 960 && owner.height == 540, "Unexpected VT working extent")
    let status = CVPixelBufferLockBaseAddress(buffer, .readOnly)
    try require(status == kCVReturnSuccess, "Cannot lock completed working buffer: \(status)")
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let address = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer))
    let rowBytes = owner.width * 8, stride = CVPixelBufferGetBytesPerRow(buffer)
    try require(stride >= rowBytes, "Invalid working row stride")
    var data = Data(count: rowBytes * owner.height)
    data.withUnsafeMutableBytes { raw in
      for y in 0..<owner.height {
        raw.baseAddress!.advanced(by: y * rowBytes).copyMemory(from: address.advanced(by: y * stride), byteCount: rowBytes)
      }
    }
    return data
  }
  private func stageRecord(_ trace: NativeFlowStageObservation, outerSeconds: Double) throws -> [String: Any] {
    try require(trace.completed && trace.entryTicks <= trace.packStartTicks && trace.packStartTicks <= trace.packEndTicks &&
      trace.packEndTicks <= trace.resizeStartTicks && trace.resizeStartTicks <= trace.resizeEndTicks && trace.resizeEndTicks <= trace.exitTicks,
      "Missing/nonmonotonic pack/resize boundaries")
    func seconds(_ start: UInt64, _ end: UInt64) -> Double { Double(end - start) / 1e9 }
    let pack = seconds(trace.packStartTicks, trace.packEndTicks)
    let resize = seconds(trace.resizeStartTicks, trace.resizeEndTicks)
    let inner = seconds(trace.entryTicks, trace.exitTicks)
    var vt: Double?, assessment: Double?
    if trace.hasPrevious {
      try require(trace.resizeEndTicks <= trace.vtStartTicks && trace.vtStartTicks <= trace.vtEndTicks &&
        trace.vtEndTicks <= trace.assessmentStartTicks && trace.assessmentStartTicks <= trace.assessmentEndTicks &&
        trace.assessmentEndTicks <= trace.exitTicks, "Missing/nonmonotonic VT/MLX boundaries")
      vt = seconds(trace.vtStartTicks, trace.vtEndTicks)
      assessment = seconds(trace.assessmentStartTicks, trace.assessmentEndTicks)
    } else {
      try require(trace.vtStartTicks == 0 && trace.vtEndTicks == 0 && trace.assessmentStartTicks == 0 && trace.assessmentEndTicks == 0,
        "Initial frame must not invent VT or MLX observations")
    }
    if let gpu = trace.packCommandGPUSeconds { try require(gpu.isFinite && gpu >= 0, "Invalid pack GPU duration") }
    let sum = pack + resize + (vt ?? 0) + (assessment ?? 0)
    return ["packCompletedWallSeconds": pack,
      "packCommandGPUSeconds": trace.packCommandGPUSeconds as Any? ?? NSNull(),
      "ciResizeReturnWallSeconds": resize,
      "vtCallbackCompletionWallSeconds": vt as Any? ?? NSNull(),
      "motionAssessmentCompletedWallSeconds": assessment as Any? ?? NSNull(),
      "innerPrepareWallSeconds": inner, "measuredStageWallSumSeconds": sum,
      "residualWallSeconds": outerSeconds - sum, "outerMinusInnerWallSeconds": outerSeconds - inner,
      "randomAccessSubmitted": trace.randomAccessSubmitted as Any? ?? NSNull(),
      "uptimeNanoseconds": ["entry": trace.entryTicks, "packStart": trace.packStartTicks, "packEnd": trace.packEndTicks,
        "resizeStart": trace.resizeStartTicks, "resizeEnd": trace.resizeEndTicks,
        "vtStart": trace.vtStartTicks, "vtEnd": trace.vtEndTicks,
        "assessmentStart": trace.assessmentStartTicks, "assessmentEnd": trace.assessmentEndTicks, "exit": trace.exitTicks]]
  }

  func testMatchedVideoToolboxStageAttribution() async throws {
    let environment = ProcessInfo.processInfo.environment
    guard let outputPath = environment["MLXDLSS_FLOW_STAGE_ATTRIBUTION_OUTPUT"] else {
      throw XCTSkip("Opt-in actual VideoToolbox stage attribution")
    }
    #if DEBUG
    throw NSError(domain: "NativeFlowStageAttribution", code: 2,
      userInfo: [NSLocalizedDescriptionKey: "Attribution benchmark requires release configuration"])
    #endif
    let inputPath = try XCTUnwrap(environment["MLXDLSS_FLOW_STAGE_INPUTS"], "Opt-in benchmark requires pinned inputs")
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { root.deleteLastPathComponent() }
    let inputURL = URL(fileURLWithPath: inputPath), output = URL(fileURLWithPath: outputPath)
    try require(!FileManager.default.fileExists(atPath: output.path) &&
      (try? output.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true, "Output must be a new path")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    var report: [String: Any] = ["schemaVersion": 1, "passed": false, "modelLoaded": false,
      "scope": "Completed NativeOpticalFlow.prepare stage attribution against unchanged production; no optimization or Live qualification",
      "pairCount": 4, "traversalsPerArm": 3, "framesPerTraversal": 12, "warmupCallsPerTraversal": 3,
      "requestedBackend": MediaMotion.videoToolbox.rawValue,
      "timingSemantics": [
        "prepareCompletedWallSeconds": "Same external call boundary in both arms, including actor scheduling and completed prepare return",
        "packCompletedWallSeconds": "Existing halfWriter.write including actor hop, pool acquisition, Metal submission and awaited command completion",
        "packCommandGPUSeconds": "Nested existing Metal command GPU end minus start; never added to wall-stage sum",
        "ciResizeReturnWallSeconds": "Legacy CIContext CVPixelBuffer render returns after completion; includes actor hop, image/colour transforms, allocation and render wait, not isolated CI GPU time",
        "vtCallbackCompletionWallSeconds": "Session.process wall including destination allocation, parameters, processing-completion callback and continuation scheduling; not isolated VT GPU time",
        "motionAssessmentCompletedWallSeconds": "MLXVideoMotion initialization including flow import, assessment, erosion, joint eval and scalar readback",
        "residualWallSeconds": "External total minus four available stage walls, signed and unclamped; nested pack GPU time excluded",
        "initialFrame": "Actual pack and resize measured; VT and motion stages are null because no prior frame exists"],
      "coreImageCompletionSource": "https://devstreaming-cdn.apple.com/videos/wwdc/2017/510lf4jlju5s1/510/510_advances_in_core_image_filters_metal_vision_and_more.pdf?dl=1",
      "coreImageCompletionPages": "89–91: legacy IOSurface/CVPixelBuffer APIs return when rendering completes",
      "limitations": ["Production wrapper is compiled in DLSSMedia; renamed attributed wrapper is compiled in XCTest. Total timing differences combine instrumentation, asymmetric untimed working-buffer readback/RAM retention and module conditions, not isolated clock overhead or optimization benefit.",
        "No added synchronization in timed flow. Existing algorithm, conversions, allocations, revision, owners and reset policy retained.",
        "Production working buffers are private. Untimed independent conversion controls use the same writer/resize primitives, not production-private pixel capture.",
        "Direct working-buffer readback and all motion hash/byte checks occur outside timing but influence operating conditions.",
        "The 1496–1507 subset repeats; wrap is a discontinuity, not continuous source-rate playback.",
        "No model, decoder, player, physical timing, broad motion-quality, optimization, Live or M5 claim."]]
    var records: [[String: Any]] = [], comparisons: [[String: Any]] = [], workingControls: [[String: Any]] = []
    var frozen: [String: String] = [:]
    func publish() throws {
      var value = report
      value["samples"] = records; value["comparisons"] = comparisons; value["workingPixelControls"] = workingControls
      try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        .write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    let priorCache = MLXRuntimeDiagnostics.cacheLimitBytes
    report["priorMLXCachePolicyBytes"] = priorCache
    defer {
      do { try MLXRuntimeDiagnostics.setCacheLimitBytes(priorCache); report["restoredMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes }
      catch { report["passed"] = false; report["cacheRestorationError"] = error.localizedDescription }
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
      if #available(macOS 15.4, *) { try require(VTOpticalFlowConfiguration.isSupported, "Actual VT unavailable; no fallback or zero-work benchmark") }
      else { try require(false, "Actual VT requires macOS15.4") }
      let inputData = try Data(contentsOf: inputURL)
      try require(digest(inputData) == "3c376d69b898a7fa0ee2474e233438c9071dbe3d63b3cece57d9b8e1bfa68181", "Input manifest pin mismatch")
      let input = try JSONDecoder().decode(Input.self, from: inputData)
      try require(input.width == 1920 && input.height == 1080 && input.frames.map(\.sourceFrameIndex) == Array(1496...1507), "Unexpected input inventory")
      let flowPath = "vendor/MLX-DLSS/Sources/DLSSMedia/NativeOpticalFlow.swift"
      let copyPath = "vendor/MLX-DLSS/Tests/DLSSMediaTests/AttributedNativeOpticalFlow.swift"
      let original = try Data(contentsOf: root.appendingPathComponent(flowPath))
      try require(digest(original) == originalFlowSHA, "Current production flow differs from frozen attribution base")
      let attributedSource = try String(contentsOf: root.appendingPathComponent(copyPath), encoding: .utf8)
      let marker = try NSRegularExpression(pattern: #"(?ms)^[ \t]*// FLOW-ATTRIBUTION-BEGIN\n.*?^[ \t]*// FLOW-ATTRIBUTION-END\n"#)
      let stripped = marker.stringByReplacingMatches(in: attributedSource, range: NSRange(attributedSource.startIndex..., in: attributedSource), withTemplate: "")
        .replacingOccurrences(of: "AttributedNativeOpticalFlow", with: "NativeOpticalFlow")
        .replacingOccurrences(of: "AttributedVideoToolboxFlowSession", with: "VideoToolboxFlowSession")
      try require(Data(stripped.utf8) == original, "Attribution must reverse byte-for-byte to production flow")
      report["reversedProductionFlowSHA256"] = originalFlowSHA
      report["copyTransformation"] = "Remove complete FLOW-ATTRIBUTION marker blocks, then reverse the two NativeOpticalFlow/VideoToolboxFlowSession type renames"
      let paths = [flowPath, copyPath, "vendor/MLX-DLSS/Tests/DLSSMediaTests/NativeFlowStageAttributionTests.swift",
        "vendor/MLX-DLSS/Sources/DLSSMedia/NativeImageIO.swift", "vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoFrame.swift",
        "vendor/MLX-DLSS/Sources/DLSSMLX/MLXVideoMotion.swift", "vendor/MLX-DLSS/Sources/DLSSMLX/MLXRuntimeDiagnostics.swift",
        ".build/debug/hdr-benchmark", ".build/debug/libFrameEngineShared.dylib", ".build/debug/mlx.metallib",
        "vendor/MLX-DLSS/.build/release/mlx.metallib"]
      for path in paths { frozen[root.appendingPathComponent(path).path] = try fileDigest(root.appendingPathComponent(path)) }
      let executable = try XCTUnwrap(Bundle(for: Self.self).executableURL)
      frozen[executable.path] = try fileDigest(executable); frozen[inputURL.path] = digest(inputData)
      report["actualTestExecutable"] = executable.path
      report["inputManifestPath"] = inputURL.path; report["inputManifestSHA256"] = digest(inputData)
      for pin in [input.captureManifest, input.source] {
        let url = pin.path.hasPrefix("/") ? URL(fileURLWithPath: pin.path) : root.appendingPathComponent(pin.path)
        try require(url.resourceValues(forKeys: [.fileSizeKey]).fileSize == pin.bytes && fileDigest(url) == pin.sha256, "Source/capture provenance differs")
        frozen[url.path] = pin.sha256
      }
      try MLXRuntimeDiagnostics.setCacheLimitBytes(256 * 1024 * 1024)
      report["benchmarkMLXCachePolicyBytes"] = MLXRuntimeDiagnostics.cacheLimitBytes
      var frames: [MLXVideoFrame] = []
      for frame in input.frames {
        let url = frame.path.hasPrefix("/") ? URL(fileURLWithPath: frame.path) : root.appendingPathComponent(frame.path)
        let bytes = try Data(contentsOf: url)
        try require(bytes.count == frame.bytes && bytes.count == 1920 * 1080 * 12 && digest(bytes) == frame.sha256, "Proxy payload pin mismatch")
        try require(frame.duration.value == 1001 && frame.duration.timescale == 24000 && frame.pts.timescale == 24000, "Unexpected source timing")
        frozen[url.path] = frame.sha256
        frames.append(try MLXVideoFrame(rgb: bytes, width: input.width, height: input.height))
      }
      report["frozenSHA256Before"] = frozen
      let production = try NativeOpticalFlow(width: input.width, height: input.height, mode: .videoToolbox)
      let attributed = try AttributedNativeOpticalFlow(width: input.width, height: input.height, mode: .videoToolbox)
      try require(production.backend == MediaMotion.videoToolbox.rawValue && attributed.backend == MediaMotion.videoToolbox.rawValue, "Actual backend must be VT")
      report["actualBackends"] = ["production": production.backend, "attributed": attributed.backend]
      var lastIndex: [String: Int] = [:], firstWorking: [Int: Data] = [:], firstWorkingSHA: [Int: String] = [:]
      var firstWorkingFiles: [Int: [String: Any]] = [:]
      var exact = true, directChecks = 0, workingHashChecks = 0
      try publish()
      for pair in 0..<4 {
        let order = pair.isMultiple(of: 2) ? ["production", "attributed"] : ["attributed", "production"]
        var reference: [[String: Any]] = [], directReferences: [Int: AttributedFlowMotionSnapshot] = [:]
        for (caseOrder, arm) in order.enumerated() {
          for traversal in 0..<3 {
            for index in frames.indices {
              let source = input.frames[index], ordinal = traversal * frames.count + index
              let wrapped = lastIndex[arm].map { $0 + 1 != source.sourceFrameIndex } ?? false
              let begin = DispatchTime.now().uptimeNanoseconds
              let motion = arm == "production"
                ? try await production.prepare(frames[index], index: source.sourceFrameIndex, sceneCutThreshold: 0.3)
                : try await attributed.prepare(frames[index], index: source.sourceFrameIndex, sceneCutThreshold: 0.3)
              let end = DispatchTime.now().uptimeNanoseconds
              let elapsed = Double(end - begin) / 1e9
              let memory = MLXRuntimeDiagnostics.memorySnapshot()
              lastIndex[arm] = source.sourceFrameIndex
              var sample: [String: Any] = ["pair": pair, "caseOrder": caseOrder, "arm": arm,
                "traversal": traversal, "sourceFrameIndex": source.sourceFrameIndex,
                "pts": ["value": source.pts.value, "timescale": Int64(source.pts.timescale)],
                "duration": ["value": source.duration.value, "timescale": Int64(source.duration.timescale)],
                "inputSHA256": source.sha256, "wrapDiscontinuity": wrapped, "excludedWarmup": index < 3,
                "prepareCompletedWallSeconds": elapsed, "outerStartUptimeNanoseconds": begin, "outerEndUptimeNanoseconds": end,
                "hasMotion": motion != nil,
                "mlxBytes": ["active": memory.activeBytes, "cache": memory.cacheBytes, "peakActive": memory.peakActiveBytes]]
              if arm == "attributed" {
                let observation = await attributed.observation()
                let trace = try XCTUnwrap(observation, "Missing completed attribution")
                try require(trace.hasPrevious == (motion != nil), "Stage history inventory mismatch")
                sample["stages"] = try stageRecord(trace, outerSeconds: elapsed)
                let pixels = try XCTUnwrap(trace.workingPixels, "Missing current working buffer")
                let bytes = try workingBytes(pixels), sha = digest(bytes)
                if let prior = firstWorkingSHA[source.sourceFrameIndex] {
                  try require(sha == prior, "Same input produced different working pixels")
                } else {
                  firstWorking[source.sourceFrameIndex] = bytes; firstWorkingSHA[source.sourceFrameIndex] = sha
                  let name = "working-frame-\(source.sourceFrameIndex).rgba16f"
                  firstWorkingFiles[source.sourceFrameIndex] = ["sourceFrameIndex": source.sourceFrameIndex,
                    "pts": ["value": source.pts.value, "timescale": Int64(source.pts.timescale)],
                    "duration": ["value": source.duration.value, "timescale": Int64(source.duration.timescale)],
                    "inputSHA256": source.sha256, "path": name, "bytes": bytes.count, "sha256": sha,
                    "width": pixels.width, "height": pixels.height, "rowBytes": pixels.width * 8,
                    "layout": "Tightly packed RGBA Float16, native little-endian; original CV row padding excluded",
                    "scope": "First completed attributed working buffer captured outside prepare timing; file saved after all timed calls"]
                }
                sample["workingPixels"] = ["sha256": sha, "logicalBytes": bytes.count, "width": pixels.width,
                  "height": pixels.height, "bytesPerRow": CVPixelBufferGetBytesPerRow(pixels.buffer),
                  "pixelFormat": CVPixelBufferGetPixelFormatType(pixels.buffer), "paddingExcluded": true]
                workingHashChecks += 1
              }
              var signature: [String: Any] = ["hasMotion": motion != nil]
              if let motion {
                let snapshot = AttributedFlowMotionSnapshot(motion)
                try require(snapshot.finite, "Nonfinite motion at \(source.sourceFrameIndex)")
                signature = ["hasMotion": true, "vectorsSHA256": digest(snapshot.vectors), "confidenceSHA256": digest(snapshot.confidence),
                  "reset": snapshot.reset, "reliableFractionBits": snapshot.reliableFraction.bitPattern,
                  "warpedLumaErrorBits": snapshot.warpedLumaError.bitPattern]
                sample["vectors"] = arraySummary(snapshot.vectors); sample["confidence"] = arraySummary(snapshot.confidence)
                sample["reset"] = snapshot.reset; sample["reliableFraction"] = snapshot.reliableFraction; sample["warpedLumaError"] = snapshot.warpedLumaError
                if index == 2 || index == 3 {
                  if caseOrder == 0 { directReferences[ordinal] = snapshot }
                  else {
                    let matches = directReferences[ordinal] == snapshot
                    sample["directBytesEqual"] = matches; directChecks += 1; exact = exact && matches
                  }
                }
              }
              sample["signature"] = signature
              if caseOrder == 0 { reference.append(signature) }
              else {
                let matches = NSDictionary(dictionary: reference[ordinal]).isEqual(to: signature)
                sample["matchedExact"] = matches; exact = exact && matches
                comparisons.append(["pair": pair, "traversal": traversal, "sourceFrameIndex": source.sourceFrameIndex,
                  "firstArm": order[0], "secondArm": arm, "firstSignature": reference[ordinal], "secondSignature": signature,
                  "matchedExact": matches, "directBytesEqual": sample["directBytesEqual"] ?? NSNull()])
              }
              records.append(sample)
            }
            report["allComparedOutputsExact"] = exact
            try publish()
          }
        }
      }
      // Persist the already-held references only after all 288 timed calls.
      // Publish each pin after its file exists, including on a later write failure.
      var savedWorkingFiles: [[String: Any]] = []
      for source in input.frames {
        let pin = try XCTUnwrap(firstWorkingFiles[source.sourceFrameIndex])
        let bytes = try XCTUnwrap(firstWorking[source.sourceFrameIndex])
        let name = try XCTUnwrap(pin["path"] as? String)
        try bytes.write(to: output.appendingPathComponent(name), options: .withoutOverwriting)
        savedWorkingFiles.append(pin)
        report["retainedWorkingPixels"] = savedWorkingFiles
        try publish()
      }
      // Independent primitive controls AFTER every timed call. These are not
      // observations of the inaccessible production estimator's private buffers.
      let writer = try MLXPixelBufferWriter(width: input.width, height: input.height, halfOutput: true)
      let imageIO = try NativeImageIO()
      for (index, frame) in frames.enumerated() {
        let packed = try await writer.write(frame)
        let working = try await imageIO.resize(packed, width: 960, height: 540)
        let bytes = try workingBytes(working), source = input.frames[index]
        let expected = try XCTUnwrap(firstWorking[source.sourceFrameIndex])
        let equal = bytes == expected
        workingControls.append(["sourceFrameIndex": source.sourceFrameIndex, "attributedSHA256": digest(expected),
          "primitiveReferenceSHA256": digest(bytes), "logicalBytes": bytes.count, "directBytesEqual": equal,
          "retainedAttributedReference": firstWorkingFiles[source.sourceFrameIndex] as Any? ?? NSNull(),
          "scope": "Untimed existing writer+CI resize primitives, after all timed calls; production-private pixels are not exposed"])
        exact = exact && equal
        try publish()
      }
      for (frame, source) in zip(frames, input.frames) {
        try require(digest(frame.copyRGBData()) == source.sha256, "Retained original proxy frame changed")
        let pin = try XCTUnwrap(firstWorkingFiles[source.sourceFrameIndex])
        let url = output.appendingPathComponent(try XCTUnwrap(pin["path"] as? String))
        try require(fileDigest(url) == firstWorkingSHA[source.sourceFrameIndex] &&
          url.resourceValues(forKeys: [.fileSizeKey]).fileSize == 960 * 540 * 8, "Retained working pixel file changed")
      }
      for (path, expected) in frozen { try require(fileDigest(URL(fileURLWithPath: path)) == expected, "Frozen source/input/runtime changed: \(path)") }
      report["allComparedOutputsExact"] = exact; report["directMotionByteChecks"] = directChecks
      report["workingPixelHashChecks"] = workingHashChecks; report["inputFramesUnchanged"] = true
      report["retainedWorkingPixelFilesUnchangedAfterWrite"] = true
      report["passed"] = exact && records.count == 288 && comparisons.count == 144 && directChecks == 24 &&
        workingHashChecks == 144 && workingControls.count == 12 && firstWorkingFiles.count == 12
      try publish()
      try require(report["passed"] as? Bool == true, "Attribution numerical comparison failed; all records retained")
    } catch {
      report["failure"] = error.localizedDescription
      try? publish()
      throw error
    }
  }
}
