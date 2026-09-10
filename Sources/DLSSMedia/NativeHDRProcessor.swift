import CoreMedia
import DLSSCore
import DLSSMLX
import Foundation

public struct NativeHDRProcessingConfiguration: Sendable {
  public var modelURL: URL?
  public var processingWidth: Int?
  public var processingHeight: Int?
  public var strength: Float
  public var colorStrength: Float
  public var maximumLuminanceRatio: Float
  public var temporal: Bool
  public var motion: MediaMotion
  public var precision: MLXComputePrecision
  public var sceneCutThreshold: Float
  public init(modelURL: URL? = nil, processingWidth: Int? = nil, processingHeight: Int? = nil,
    strength: Float = 1, colorStrength: Float = 1, maximumLuminanceRatio: Float = 2,
    temporal: Bool = true, motion: MediaMotion = .automatic,
    precision: MLXComputePrecision = .float16, sceneCutThreshold: Float = 0.3) {
    self.modelURL = modelURL; self.processingWidth = processingWidth; self.processingHeight = processingHeight
    self.strength = strength; self.colorStrength = colorStrength; self.maximumLuminanceRatio = maximumLuminanceRatio
    self.temporal = temporal; self.motion = motion; self.precision = precision; self.sceneCutThreshold = sceneCutThreshold
  }
  func validate() throws {
    guard strength.isFinite, (0...1).contains(strength), colorStrength.isFinite, (0...1).contains(colorStrength),
      maximumLuminanceRatio.isFinite, maximumLuminanceRatio >= 1,
      sceneCutThreshold.isFinite, (0...1).contains(sceneCutThreshold),
      (processingWidth == nil) == (processingHeight == nil),
      processingWidth.map({ (1...16384).contains($0) }) ?? true,
      processingHeight.map({ (1...16384).contains($0) }) ?? true else { throw MLXMediaError("Invalid native HDR processing policy") }
  }
}

/// Every view refers to one retained source frame and exact rational timestamp.
/// Original, identity and enhanced are linear BT.2020 nits; proxy is bounded sRGB.
/// Display mapping belongs to the presenter and must run once after this result.
public struct NativeHDRStageTimings: Sendable {
  /// Completed stage wall time, including GPU waits; not isolated GPU duration.
  public var proxySeconds: Double = 0
  public var motionSeconds: Double = 0
  public var inferenceSeconds: Double = 0
  public var reconstructionSeconds: Double = 0
}

public struct NativeHDRProcessedFrame: Sendable {
  public let source: MLXHDRFrame
  public let proxy: MLXVideoFrame
  public let identity: MLXVideoFrame
  public let enhanced: MLXVideoFrame
  public let usedModel: Bool
  public let historyReset: Bool
  public let timings: NativeHDRStageTimings
  public var original: MLXVideoFrame { source.original }
  public var metadata: MLXHDRFrameMetadata { source.metadata }
}

/// Persistent model and motion resources surrounding an SDR-only neural proxy.
/// Calls must be sequential. A source, generation, geometry or PTS discontinuity
/// resets history; presenting a completed result again does not call this actor.
public actor NativeHDRProcessor {
  private var configuration: NativeHDRProcessingConfiguration
  private let renderer: MLXNeuralRenderingDeviceTemporalBackend?
  private let codec = MLXNeuralRenderingDisplayCodec()
  private var flow: NativeOpticalFlow?
  private var previous: MLXHDRFrameMetadata?
  private var dimensions: (Int, Int)?
  private var busy = false
  private var resetRequested = false

  public init(configuration: NativeHDRProcessingConfiguration = .init()) throws {
    try configuration.validate()
    self.configuration = configuration
    renderer = try configuration.modelURL.map { try MLXNeuralRenderingDeviceTemporalBackend(
      packageURL: $0, executionMode: .metalFused, computePrecision: configuration.precision) }
  }

#if MLXDLSS_TEMPORAL_DIAGNOSTICS
  public func enableTemporalDiagnostics(frameIndices: [UInt64]) async throws {
    guard !busy, previous == nil, let renderer else {
      throw MLXMediaError("Arm temporal diagnostics before native HDR processing")
    }
    busy = true
    defer { busy = false }
    try await renderer.enableTemporalDiagnostics(frameIndices: frameIndices)
  }

  public func temporalDiagnosticSnapshots() async throws -> [MLXTemporalDiagnosticSnapshot] {
    guard !busy, let renderer else { throw MLXMediaError("Read temporal diagnostics between frames") }
    busy = true
    defer { busy = false }
    return try await renderer.temporalDiagnosticSnapshots()
  }

  public func temporalDiagnosticState() async throws -> MLXTemporalDiagnosticState {
    guard !busy, let renderer else { throw MLXMediaError("Read temporal state between frames") }
    busy = true
    defer { busy = false }
    return try await renderer.temporalDiagnosticState()
  }

  public func replayTemporalDiagnostic(baseFrameIndex: UInt64, noiseFromFrameIndex: UInt64,
    historyFromFrameIndex: UInt64) async throws -> MLXTemporalDiagnosticReplay {
    guard !busy, !resetRequested, let renderer else { throw MLXMediaError("Replay between completed native frames") }
    busy = true
    defer { busy = false }
    let replay = try await renderer.replayTemporalDiagnostic(baseFrameIndex: baseFrameIndex,
      noiseFromFrameIndex: noiseFromFrameIndex, historyFromFrameIndex: historyFromFrameIndex)
    if resetRequested { throw CancellationError() }
    return replay
  }
#endif
  /// Strength changes preserve temporal inputs. Processing-size changes reset
  /// history before the next frame and do not reload the model weights.
  public func configure(strength: Float, colorStrength: Float, processingWidth: Int? = nil,
    processingHeight: Int? = nil) async throws {
    guard !busy else { throw MLXMediaError("Configure HDR processing between submissions") }
    busy = true
    defer { busy = false }
    var updated = configuration
    updated.strength = strength; updated.colorStrength = colorStrength
    updated.processingWidth = processingWidth; updated.processingHeight = processingHeight
    try updated.validate()
    if updated.processingWidth != configuration.processingWidth || updated.processingHeight != configuration.processingHeight {
      await clearHistory()
    }
    configuration = updated
    resetRequested = false
  }

  private func seconds(since start: ContinuousClock.Instant) -> Double {
    let value = start.duration(to: .now).components
    return Double(value.seconds) + Double(value.attoseconds) / 1e18
  }

  public func reset() async {
    if busy { resetRequested = true; return }
    busy = true
    defer { busy = false }
    await clearHistory()
    resetRequested = false
  }

  private func clearHistory() async {
    await renderer?.reset(sequenceID: nil)
    flow = nil; previous = nil; dimensions = nil
  }

  public func process(_ frame: MLXHDRFrame) async throws -> NativeHDRProcessedFrame {
    guard !busy else { throw MLXMediaError("Submit native HDR frames sequentially") }
    busy = true
    defer { busy = false }
    do {
      let result = try await processFrame(frame)
      if resetRequested { throw CancellationError() }
      return result
    } catch {
      await clearHistory()
      resetRequested = false
      throw error
    }
  }

  private func processFrame(_ frame: MLXHDRFrame) async throws -> NativeHDRProcessedFrame {
    try Task.checkCancellation()
    let metadata = frame.metadata
    let width = frame.original.width, height = frame.original.height
    let discontinuity = previous.map {
      $0.sourceID != metadata.sourceID || $0.streamID != metadata.streamID || $0.generation != metadata.generation
        || $0.frameIndex &+ 1 != metadata.frameIndex || metadata.time <= $0.time
        || abs((metadata.time - ($0.time + $0.duration)).seconds) > max(0.001, $0.duration.seconds * 0.5)
        || $0.color != metadata.color
    } ?? true
    let resetHistory = discontinuity || dimensions?.0 != width || dimensions?.1 != height
    if resetHistory {
      await renderer?.reset(sequenceID: nil)
      flow = configuration.temporal && renderer != nil
        ? try NativeOpticalFlow(width: width, height: height, mode: configuration.motion) : nil
    }
    let color = NeuralRenderingDisplayCodecConfiguration(whitePoint: metadata.color.referenceWhiteNits,
      transferStrength: configuration.strength, colorStrength: configuration.colorStrength,
      maximumLuminanceRatio: configuration.maximumLuminanceRatio, workingPrimaries: .bt2020)
    var timings = NativeHDRStageTimings()
    var stage = ContinuousClock.now
    let proxy = codec.encode(frame.original, configuration: color)
    timings.proxySeconds = seconds(since: stage)
    stage = .now
    let identity = codec.resolve(proxy: proxy, model: proxy, original: frame.original, configuration: color)
    timings.reconstructionSeconds = seconds(since: stage)
    var enhanced = frame.original
    var sceneCut = false
    if let renderer, configuration.strength > 0 {
      stage = .now
      let motion = try await flow?.prepare(proxy, index: Int(truncatingIfNeeded: metadata.frameIndex),
        sceneCutThreshold: configuration.sceneCutThreshold)
      timings.motionSeconds = seconds(since: stage)
      stage = .now
      sceneCut = motion?.reset == true
      let model = try await renderer.renderVideoFrame(proxy, motion: motion,
        context: .init(streamID: metadata.streamID, frameIndex: metadata.frameIndex,
          discontinuity: resetHistory ? .explicit : nil), temporal: configuration.temporal,
        outputOptions: try .init(width: width, height: height),
        processingWidth: configuration.processingWidth, processingHeight: configuration.processingHeight,
        processingInputRange: .boundedSRGB)
      timings.inferenceSeconds = seconds(since: stage)
      stage = .now
      enhanced = codec.resolve(proxy: proxy, model: model, original: frame.original, configuration: color)
      timings.reconstructionSeconds += seconds(since: stage)
    } else if configuration.strength == 0 {
      // Skipped model input cannot be a temporal predecessor when re-enabled.
      await renderer?.reset(sequenceID: nil)
      flow = nil
    }
    try Task.checkCancellation()
    previous = configuration.strength == 0 ? nil : metadata
    dimensions = (width, height)
    return NativeHDRProcessedFrame(source: frame, proxy: proxy, identity: identity, enhanced: enhanced,
      usedModel: renderer != nil && configuration.strength > 0, historyReset: resetHistory || sceneCut, timings: timings)
  }
}
