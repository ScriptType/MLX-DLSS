import AVFoundation
import CoreVideo
import DLSSMLX
import Foundation

struct NativeDecodedFrame: Sendable {
  let rgb: MLXVideoFrame
  let time: CMTime
  let duration: CMTime
}

/// SDR-export adapter around the retained-original HDR decoder. The file writer
/// is explicitly SDR; playback uses NativeHDRVideoReader/NativeHDRProcessor.
@available(macOS 26.0, *)
actor NativeVideoReader {
  nonisolated let estimatedFrames: Int
  nonisolated let nominalFrameRate: Float
  private let reader: NativeHDRVideoReader
  private let imageIO: NativeImageIO
  private let codec = MLXNeuralRenderingDisplayCodec()
  private let options: MediaProcessingOptions
  private var emitted = 0
  private var skipped = 0

  init(url: URL, options: MediaProcessingOptions, timeRange: CMTimeRange? = nil) async throws {
    reader = try await NativeHDRVideoReader(url: url, timeRange: timeRange)
    nominalFrameRate = reader.nominalFrameRate
    estimatedFrames = min(options.frameLimit ?? Int.max, max(0, reader.estimatedFrames - options.startFrame))
    imageIO = try NativeImageIO()
    self.options = options
  }

  func next() async throws -> NativeDecodedFrame? {
    if let limit = options.frameLimit, emitted >= limit { await reader.cancel(); return nil }
    while let frame = try await reader.next() {
      if skipped < options.startFrame { skipped += 1; continue }
      var rgb = codec.encode(frame.original, configuration: .init(
        whitePoint: frame.metadata.color.referenceWhiteNits, workingPrimaries: .bt2020))
      if !frame.metadata.transform.isIdentity {
        let writer = try MLXPixelBufferWriter(width: rgb.width, height: rgb.height, halfOutput: true)
        let pixels = try await writer.write(rgb)
        rgb = try await MLXVideoFrame(pixelBuffer: imageIO.convert(pixels, transform: frame.metadata.transform))
      }
      emitted += 1
      return NativeDecodedFrame(rgb: rgb, time: frame.metadata.time, duration: frame.metadata.duration)
    }
    return nil
  }

  func cancel() async { await reader.cancel() }
}

/// The audio producer advances independently from video under encoder backpressure.
/// Composition time scaling plus the spectral algorithm preserves pitch in slow motion.
@available(macOS 26.0, *)
actor NativeAudioReader {
  private let reader: AVAssetReader
  private let provider: AVAssetReaderOutput.Provider<CMReadySampleBuffer<CMSampleBuffer.DynamicContent>>
  private var pending: CMReadySampleBuffer<CMSampleBuffer.DynamicContent>?
  private var endTime: CMTime
  private let paddingURL: URL?
  private var done = false

  static func open(url: URL, start: CMTime, timeScale: Int) async throws -> NativeAudioReader? {
    let asset = AVURLAsset(url: url)
    let tracks = try await asset.loadTracks(withMediaType: .audio)
    guard !tracks.isEmpty else { return nil }
    let duration = try await asset.load(.duration)
    let sourceDuration = duration - start
    guard sourceDuration > .zero else { return nil }
    let paddingURL = timeScale == 1 ? nil : try makeSilence()
    var retainedPadding = false
    defer { if !retainedPadding, let paddingURL { try? FileManager.default.removeItem(at: paddingURL) } }
    let paddingAsset = paddingURL.map { AVURLAsset(url: $0) }
    let paddingTrack = try await paddingAsset?.loadTracks(withMediaType: .audio).first
    let composition = AVMutableComposition()
    for source in tracks {
      guard let track = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else {
        throw MLXMediaError("Cannot prepare source audio")
      }
      let available = try await source.load(.timeRange)
      let requested = CMTimeRange(start: start, end: duration)
      let range = CMTimeRangeGetIntersection(available, otherRange: requested)
      if range.duration > .zero {
        try track.insertTimeRange(range, of: source, at: range.start - start)
      }
      if let paddingTrack {
        try track.insertTimeRange(CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 1)),
          of: paddingTrack, at: sourceDuration)
      }
    }
    if timeScale != 1 {
      let paddedDuration = sourceDuration + CMTime(value: 1, timescale: 1)
      composition.scaleTimeRange(CMTimeRange(start: .zero, duration: paddedDuration),
        toDuration: CMTimeMultiply(paddedDuration, multiplier: Int32(timeScale)))
    }
    let result = try NativeAudioReader(composition: composition,
      endTime: CMTimeMultiply(sourceDuration, multiplier: Int32(timeScale)), paddingURL: paddingURL)
    retainedPadding = true
    return result
  }

  private init(composition: AVMutableComposition, endTime: CMTime, paddingURL: URL?) throws {
    self.endTime = endTime
    self.paddingURL = paddingURL
    reader = try AVAssetReader(asset: composition)
    let output = AVAssetReaderAudioMixOutput(audioTracks: composition.tracks, audioSettings: [
      AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 48000, AVNumberOfChannelsKey: 2,
      AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsNonInterleaved: false,
    ])
    output.audioTimePitchAlgorithm = .spectral
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw MLXMediaError("Cannot decode source audio") }
    provider = reader.outputProvider(for: output)
    guard reader.startReading() else { throw reader.error ?? MLXMediaError("Cannot start audio decoder") }
  }

  func next(upTo time: CMTime) async throws -> CMReadySampleBuffer<CMSampleBuffer.DynamicContent>? {
    try Task.checkCancellation()
    guard !done else { return nil }
    if pending == nil { pending = try await provider.next() }
    guard let sample = pending else {
      if reader.status == .failed { throw reader.error ?? MLXMediaError("Audio decoding failed") }
      done = true
      return nil
    }
    guard sample.presentationTimeStamp < endTime else { done = true; pending = nil; return nil }
    guard sample.presentationTimeStamp < time else { return nil }
    pending = nil
    if sample.presentationTimeStamp + sample.duration > endTime {
      done = true
      let count = min(sample.sampleCount, Int(((endTime - sample.presentationTimeStamp).seconds * 48000).rounded()))
      guard count > 0 else { return nil }
      return try sample.withUnsafeSampleBuffer {
        // The new CF object retains immutable PCM from the ready sample. Swift
        // cannot infer ownership through Core Media's shallow-copy initializer.
        nonisolated(unsafe) let copy = try CMSampleBuffer(copying: $0, forRange: 0..<count)
        return CMReadySampleBuffer(unsafeBuffer: copy)
      }
    }
    return sample
  }

  func limit(to time: CMTime) { endTime = min(endTime, time) }

  func cancel() { reader.cancelReading(); pending = nil; done = true }

  deinit { if let paddingURL { try? FileManager.default.removeItem(at: paddingURL) } }

  private static func makeSilence() throws -> URL {
    // Scaled AVAssetReader edits omit the stretcher's buffered tail at EOF.
    // A real PCM segment flushes it; an empty composition range does not. The
    // reader clips output to the original duration and removes this private file.
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxdlss-audio-\(UUID().uuidString).wav")
    guard let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2),
      let silence = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 48000) else {
      throw MLXMediaError("Cannot prepare audio tail")
    }
    silence.frameLength = 48000
    for channel in 0..<2 { silence.floatChannelData![channel].initialize(repeating: 0, count: 48000) }
    do {
      var settings = format.settings
      settings[AVLinearPCMIsNonInterleaved] = false
      let file = try AVAudioFile(forWriting: url, settings: settings)
      try file.write(from: silence)
      return url
    } catch { try? FileManager.default.removeItem(at: url); throw error }
  }
}
