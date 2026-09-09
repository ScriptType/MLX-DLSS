import AVFoundation
import CoreVideo
import DLSSMLX
import Foundation

/// Retained decoder planes before any GPU conversion. The producer is complete;
/// callers may import once in their own shared frame engine.
public struct NativeHDRDecodedFrame: Sendable {
  public let pixelBuffer: MLXPixelBuffer
  public let metadata: MLXHDRFrameMetadata
  public init(pixelBuffer: MLXPixelBuffer, metadata: MLXHDRFrameMetadata) {
    self.pixelBuffer = pixelBuffer; self.metadata = metadata
  }
}

/// Planar native decoder for playback and numeric captures. Geometry is retained,
/// not baked through an SDR Core Image conversion; the native presenter applies it.
@available(macOS 26.0, *)
public actor NativeHDRVideoReader {
  public nonisolated let nominalFrameRate: Float
  public nonisolated let estimatedFrames: Int
  private let reader: AVAssetReader
  private let provider: AVAssetReaderOutput.Provider<CMReadySampleBuffer<CMSampleBuffer.DynamicContent>>
  private var importer: MLXHDRImporter?
  private let sourceID: String
  private let generation: UInt64
  private let streamID: UInt64
  private let transform: CGAffineTransform
  private let referenceWhiteNits: Float
  private let hlgPeakNits: Float
  private let fallbackTags: [String: String]
  private let masteringDisplay: Data?
  private let contentLightLevel: Data?
  private var frameIndex: UInt64 = 0

  public init(url: URL, sourceID: String? = nil, generation: UInt64 = 0,
    referenceWhiteNits: Float = 203, hlgPeakNits: Float = 1000,
    timeRange: CMTimeRange? = nil) async throws {
    let asset = AVURLAsset(url: url)
    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
      throw MLXMediaError("The file contains no video track")
    }
    self.sourceID = sourceID ?? url.standardizedFileURL.absoluteString
    self.generation = generation; self.referenceWhiteNits = referenceWhiteNits; self.hlgPeakNits = hlgPeakNits
    streamID = UInt64(UInt32(bitPattern: track.trackID))
    transform = try await track.load(.preferredTransform)
    let rate = try await track.load(.nominalFrameRate)
    nominalFrameRate = rate.isFinite && rate > 0 ? rate : 30
    let duration = try await asset.load(.duration)
    let count = ceil((timeRange?.duration ?? duration).seconds * Double(nominalFrameRate))
    estimatedFrames = count.isFinite && count >= 0 && count < Double(Int.max) ? Int(count) : 0
    let descriptions = try await track.load(.formatDescriptions)
    let extensions = descriptions.first.flatMap { CMFormatDescriptionGetExtensions($0) }.map { $0 as NSDictionary } ?? [:]
    var tags: [String: String] = [:]
    for key in [kCVImageBufferTransferFunctionKey, kCVImageBufferColorPrimariesKey,
      kCVImageBufferYCbCrMatrixKey, kCVImageBufferChromaLocationTopFieldKey] {
      if let value = extensions[key] { tags[key as String] = String(describing: value) }
    }
    fallbackTags = tags
    masteringDisplay = extensions[kCVImageBufferMasteringDisplayColorVolumeKey] as? Data
    contentLightLevel = extensions[kCVImageBufferContentLightLevelInfoKey] as? Data
    let transfer = tags[kCVImageBufferTransferFunctionKey as String]
    let hdr = transfer == kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String
      || transfer == kCVImageBufferTransferFunction_ITU_R_2100_HLG as String
    let atoms = extensions[kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms] as? [String: Data]
    let hevcDepth = atoms?["hvcC"].flatMap { $0.count > 17 ? 8 + Int($0[17] & 7) : nil }
    let bitDepth = (extensions[kCMFormatDescriptionExtension_BitsPerComponent] as? NSNumber)?.intValue ?? hevcDepth ?? 8
    let useP010 = hdr || bitDepth > 8
    reader = try AVAssetReader(asset: asset)
    if let timeRange { reader.timeRange = timeRange }
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: useP010 ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:],
      kCVPixelBufferMetalCompatibilityKey as String: true,
    ])
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else { throw MLXMediaError("Cannot attach planar video decoder") }
    provider = reader.outputProvider(for: output)
    guard reader.startReading() else { throw reader.error ?? MLXMediaError("Cannot start planar video decoder") }
  }

  public func next() async throws -> MLXHDRFrame? {
    guard let decoded = try await nextDecoded() else { return nil }
    if importer == nil { importer = try MLXHDRImporter() }
    return try await importer!.importFrame(pixelBuffer: decoded.pixelBuffer, metadata: decoded.metadata)
  }

  public func nextDecoded() async throws -> NativeHDRDecodedFrame? {
    try Task.checkCancellation()
    guard let sample = try await provider.next() else {
      if reader.status == .failed { throw reader.error ?? MLXMediaError("Video decoding failed") }
      return nil
    }
    guard case .pixelBuffer(let image) = sample.content else { throw MLXMediaError("Decoder returned no image") }
    let pixels = image.withUnsafeBuffer { MLXPixelBuffer($0) }
    let time = sample.presentationTimeStamp
    guard time.isNumeric else { throw MLXMediaError("Video frame has no presentation timestamp") }
    var duration = sample.duration
    if !duration.isNumeric || duration <= .zero {
      duration = CMTime(seconds: 1 / Double(nominalFrameRate), preferredTimescale: 60000)
    }
    var color = try MLXHDRColorMetadata.read(from: pixels.buffer,
      referenceWhiteNits: referenceWhiteNits, hlgPeakNits: hlgPeakNits, fallbackTags: fallbackTags)
    color.masteringDisplay = color.masteringDisplay ?? masteringDisplay
    color.contentLightLevel = color.contentLightLevel ?? contentLightLevel
    let aspect = CVBufferCopyAttachment(pixels.buffer, kCVImageBufferPixelAspectRatioKey, nil) as? [String: NSNumber]
    let horizontal = aspect?[kCVImageBufferPixelAspectRatioHorizontalSpacingKey as String]?.doubleValue ?? 1
    let vertical = aspect?[kCVImageBufferPixelAspectRatioVerticalSpacingKey as String]?.doubleValue ?? 1
    let metadata = MLXHDRFrameMetadata(time: time, duration: duration, sourceID: sourceID,
      streamID: streamID, frameIndex: frameIndex, generation: generation,
      crop: CVImageBufferGetCleanRect(pixels.buffer), transform: transform,
      pixelAspectRatio: horizontal / vertical, color: color)
    frameIndex += 1
    return NativeHDRDecodedFrame(pixelBuffer: pixels, metadata: metadata)
  }

  public func cancel() { reader.cancelReading() }
}
