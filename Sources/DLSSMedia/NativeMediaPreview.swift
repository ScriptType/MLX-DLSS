import AVFoundation
import DLSSCore
import DLSSMLX
import Foundation

public struct MediaPreviewRequest: Equatable, Sendable {
  public let input: URL
  public let isVideo: Bool
  public let time: Double
  public let options: MediaProcessingOptions

  public init(input: URL, isVideo: Bool, time: Double = 0, options: MediaProcessingOptions) {
    self.input = input
    self.isVideo = isVideo
    self.time = time
    self.options = options
  }
}

public struct MediaPreviewResult: Sendable {
  public let original: CGImage
  public let processed: CGImage
  public let time: Double
  public let duration: Double
  public let frameInterval: Double
  public let historyFrames: Int
  public let elapsedSeconds: Double
}

/// Retains the selected source frames and loaded weights across control changes.
/// Callers submit one request at a time and discard superseded results. Preview
/// history is a bounded local window, independent of export's full sequence.
@available(macOS 26.0, *)
public actor NativeMediaPreview {
  private let io: NativeImageIO
  private var sourceKey: SourceKey?
  private var frames: [NativeDecodedFrame] = []
  private var original: CGImage?
  private var duration: Double = 0
  private var frameInterval: Double = 1 / 30
  private var renderer: MLXNeuralRenderingDeviceTemporalBackend?
  private var modelURL: URL?
  private var precision: MLXComputePrecision?
  private var upscaler: MLXNativeSuperResolver?
  private var superResolutionURL: URL?
  private var motionKey: MotionKey?
  private var motions: [MLXVideoMotion?] = []
  private var busy = false

  private struct SourceKey: Equatable {
    let url: URL
    let isVideo: Bool
    let time: Double
  }
  private struct MotionKey: Equatable {
    let mode: MediaMotion
    let threshold: Float
  }

  public init() throws { io = try NativeImageIO() }

  public func render(_ request: MediaPreviewRequest) async throws -> MediaPreviewResult {
    guard !busy else { throw MLXMediaError("Submit preview requests sequentially") }
    guard request.time.isFinite, request.time >= 0 else { throw MLXMediaError("Invalid preview time") }
    if request.options.renderingModel != nil || request.options.superResolutionWeights != nil {
      try request.options.validate()
    }
    busy = true
    defer { busy = false }
    let started = ContinuousClock.now
    let key = SourceKey(url: request.input, isVideo: request.isVideo, time: request.isVideo ? request.time : 0)
    if sourceKey != key {
      try await loadSource(key)
      sourceKey = key
      original = nil
      motionKey = nil
      motions = []
    }
    guard let selected = frames.last else { throw MLXMediaError("No frame at the selected time") }
    if original == nil { original = try await io.displayImage(selected.rgb) }
    try Task.checkCancellation()
    let options = request.options
    var result = selected.rgb
    var historyFrames = 0
    if let url = options.renderingModel {
      if renderer == nil || modelURL != url || precision != options.precision {
        renderer = nil
        renderer = try MLXNeuralRenderingDeviceTemporalBackend(packageURL: url,
          executionMode: .metalFused, computePrecision: options.precision)
        modelURL = url
        precision = options.precision
      }
      let renderer = renderer!
      let temporal = request.isVideo && options.temporal
      if temporal {
        let key = MotionKey(mode: options.motion, threshold: options.sceneCutThreshold)
        if motionKey != key {
          let flow = try NativeOpticalFlow(width: selected.rgb.width, height: selected.rgb.height, mode: options.motion)
          var prepared: [MLXVideoMotion?] = []
          for (index, frame) in frames.enumerated() {
            try Task.checkCancellation()
            prepared.append(try await flow.prepare(frame.rgb, index: index, sceneCutThreshold: options.sceneCutThreshold))
          }
          motions = prepared
          motionKey = key
        }
      }
      await renderer.reset(sequenceID: 1)
      let outputOptions = try MLXVideoOutputOptions(width: selected.rgb.width, height: selected.rgb.height,
        detailStrength: options.detailStrength, colourStrength: options.colourStrength, radius: options.detailRadius)
      let indices = temporal ? Array(frames.indices) : [frames.count - 1]
      for (ordinal, index) in indices.enumerated() {
        try Task.checkCancellation()
        result = try await renderer.renderVideoFrame(frames[index].rgb, motion: temporal ? motions[index] : nil,
          context: .init(streamID: 1, frameIndex: UInt64(ordinal)), processingScale: options.processingScale,
          temporal: temporal, outputOptions: outputOptions, featureControls: options.profile.featureControls,
          intensity: options.intensity)
      }
      try Task.checkCancellation()
      historyFrames = indices.count - 1
    }
    if let url = options.superResolutionWeights {
      if upscaler == nil || superResolutionURL != url {
        upscaler = nil
        upscaler = try MLXNativeSuperResolver(weightsURL: url)
        superResolutionURL = url
      }
      result = try await upscaler!.upscale(result)
    } else {
      upscaler = nil
      superResolutionURL = nil
    }
    try Task.checkCancellation()
    let processed = options.renderingModel != nil || options.superResolutionWeights != nil
      ? try await io.displayImage(result) : original!
    let elapsed = started.duration(to: .now).components
    return MediaPreviewResult(original: original!, processed: processed, time: selected.time.seconds,
      duration: duration, frameInterval: frameInterval, historyFrames: historyFrames,
      elapsedSeconds: Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18)
  }

  private func loadSource(_ key: SourceKey) async throws {
    if !key.isVideo {
      let image = try await io.read(key.url)
      frames = [NativeDecodedFrame(rgb: image, time: .zero, duration: .zero)]
      duration = 0
      return
    }
    let asset = AVURLAsset(url: key.url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw MLXMediaError("The file contains no video track")
    }
    let trackRange = try await track.load(.timeRange)
    let rate = try await track.load(.nominalFrameRate)
    let frameInterval = rate.isFinite && rate > 0 ? 1 / Double(rate) : 1 / 30
    guard trackRange.end.seconds.isFinite, trackRange.duration > .zero else {
      throw MLXMediaError("Video preview requires a finite duration")
    }
    let duration = trackRange.end.seconds
    let time = min(max(trackRange.start.seconds, key.time), max(trackRange.start.seconds, duration - frameInterval))
    let start = CMTime(seconds: max(trackRange.start.seconds, time - 3 * frameInterval), preferredTimescale: 60000)
    let reader = try await NativeVideoReader(url: key.url, options: MediaProcessingOptions(),
      timeRange: CMTimeRange(start: start, end: trackRange.end))
    do {
      var selected: [NativeDecodedFrame] = []
      while let frame = try await reader.next() {
        selected.append(frame)
        if selected.count > 4 { selected.removeFirst() }
        if frame.time.seconds >= time - 1e-6 { break }
      }
      await reader.cancel()
      guard !selected.isEmpty else { throw MLXMediaError("No frame at the selected time") }
      frames = selected
      self.duration = duration
      self.frameInterval = frameInterval
    } catch { await reader.cancel(); throw error }
  }
}
