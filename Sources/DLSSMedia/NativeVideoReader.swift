import AVFoundation
import CoreVideo
import DLSSMLX
import Foundation

struct NativeDecodedFrame: Sendable {
  let rgb: MLXVideoFrame
  let time: CMTime
  let duration: CMTime
}

@available(macOS 26.0, *)
actor NativeVideoReader {
  nonisolated let estimatedFrames: Int
  nonisolated let nominalFrameRate: Float
  private let reader: AVAssetReader
  private let provider: AVAssetReaderOutput.Provider<CMReadySampleBuffer<CMSampleBuffer.DynamicContent>>
  private let transform: CGAffineTransform
  private let imageIO: NativeImageIO
  private let options: MediaProcessingOptions
  private var index = 0
  private var emitted = 0

  init(url: URL, options: MediaProcessingOptions, timeRange: CMTimeRange? = nil) async throws {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw MLXMediaError("The file contains no video track")
    }
    let rate = try await track.load(.nominalFrameRate)
    let duration = try await asset.load(.duration)
    nominalFrameRate = rate.isFinite && rate > 0 ? rate : 30
    let estimate = ceil(duration.seconds * Double(nominalFrameRate))
    estimatedFrames = estimate.isFinite && estimate >= 0 && estimate < Double(Int.max)
      ? min(options.frameLimit ?? Int.max, max(0, Int(estimate) - options.startFrame)) : 0
    transform = try await track.load(.preferredTransform)
    self.options = options
    imageIO = try NativeImageIO()
    reader = try AVAssetReader(asset: asset)
    if let timeRange { reader.timeRange = timeRange }
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      kCVPixelBufferMetalCompatibilityKey as String: true,
    ])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw MLXMediaError("Cannot attach native video decoder") }
    provider = reader.outputProvider(for: output)
    guard reader.startReading() else { throw reader.error ?? MLXMediaError("Cannot start native video decoder") }
  }

  func next() async throws -> NativeDecodedFrame? {
    if let limit = options.frameLimit, emitted >= limit { reader.cancelReading(); return nil }
    while true {
      try Task.checkCancellation()
      guard let sample = try await provider.next() else {
        if reader.status == .failed { throw reader.error ?? MLXMediaError("Video decoding failed") }
        return nil
      }
      let currentIndex = index
      index += 1
      if currentIndex < options.startFrame { continue }
      guard case .pixelBuffer(let image) = sample.content else { throw MLXMediaError("Decoder returned no image") }
      let time = sample.presentationTimeStamp
      var duration = sample.duration
      guard time.isNumeric else { throw MLXMediaError("Video frame has no presentation timestamp") }
      if !duration.isNumeric || duration <= .zero {
        duration = CMTime(seconds: 1 / Double(nominalFrameRate), preferredTimescale: 60000)
      }
      let original = image.withUnsafeBuffer { MLXPixelBuffer($0) }
      let pixels = transform.isIdentity ? original : try await imageIO.convert(original, transform: transform)
      let rgb = try MLXVideoFrame(pixelBuffer: pixels)
      emitted += 1
      return NativeDecodedFrame(rgb: rgb, time: time, duration: duration)
    }
  }

  func cancel() { reader.cancelReading() }
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
