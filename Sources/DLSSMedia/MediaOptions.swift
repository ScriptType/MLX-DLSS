import DLSSCore
import DLSSMLX
import Foundation

public enum MediaEffectOrder: String, Codable, CaseIterable, Sendable {
  case renderingThenGeneration = "nr-fg"
  case generationThenRendering = "fg-nr"
}

public enum MediaVideoCodec: String, Codable, CaseIterable, Sendable {
  case h264, hevc, prores
}

public enum MediaMotion: String, Codable, CaseIterable, Sendable {
  case automatic, videoToolbox = "videotoolbox", vision, zero
}

public struct MediaProcessingOptions: Equatable, Sendable {
  public var renderingModel: URL?
  public var frameGenerationWeights: URL?
  public var superResolutionWeights: URL?
  public var dlssSuperResolutionModel: URL?
  public var order: MediaEffectOrder = .renderingThenGeneration
  public var temporal = true
  public var motion: MediaMotion = .automatic
  public var sceneCutThreshold: Float = 0.3
  public var processingScale: Float = 1
  public var detailStrength: Float = 1
  public var colourStrength: Float = 1
  public var detailRadius: Float = 4
  public var intensity: Float = 1
  public var profile: NeuralRenderingControlProfile = .standard
  public var frameGenerationFactor = 2
  public var slowMotion = false
  public var includeAudio = true
  public var codec: MediaVideoCodec = .h264
  public var bitrate: Int?
  public var startFrame = 0
  public var frameLimit: Int?
  public var precision: MLXComputePrecision = .float16

  public init(renderingModel: URL? = nil, frameGenerationWeights: URL? = nil, superResolutionWeights: URL? = nil,
    dlssSuperResolutionModel: URL? = nil) {
    self.renderingModel = renderingModel
    self.frameGenerationWeights = frameGenerationWeights
    self.superResolutionWeights = superResolutionWeights
    self.dlssSuperResolutionModel = dlssSuperResolutionModel
  }

  public func validate() throws {
    guard renderingModel != nil || frameGenerationWeights != nil || superResolutionWeights != nil || dlssSuperResolutionModel != nil else {
      throw MLXMediaError("Select neural rendering, frame generation or super resolution weights")
    }
    guard superResolutionWeights == nil || dlssSuperResolutionModel == nil else {
      throw MLXMediaError("Choose one upscaler: DLSS SR or RTX VSR")
    }
    guard processingScale.isFinite, (1...4).contains(processingScale),
      detailStrength.isFinite, (0...8).contains(detailStrength),
      colourStrength.isFinite, (0...4).contains(colourStrength),
      detailRadius.isFinite, (0.5...64).contains(detailRadius),
      intensity.isFinite, (0...2).contains(intensity),
      sceneCutThreshold.isFinite, (0...1).contains(sceneCutThreshold),
      (2...16).contains(frameGenerationFactor), startFrame >= 0,
      frameLimit.map({ $0 > 0 }) ?? true, bitrate.map({ $0 > 0 }) ?? true
    else { throw MLXMediaError("Invalid processing controls, frame range, factor or bitrate") }
    guard !slowMotion || frameGenerationWeights != nil else {
      throw MLXMediaError("Slow motion requires frame generation")
    }
  }
}

public struct MediaProgress: Sendable {
  public let inputFrames: Int
  public let outputFrames: Int
  public let estimatedInputFrames: Int
  public let sceneResets: Int
  public let elapsedSeconds: Double

  public var fraction: Double {
    estimatedInputFrames > 0 ? min(1, Double(inputFrames) / Double(estimatedInputFrames)) : 0
  }
}

public struct MediaProcessingResult: Codable, Sendable {
  public let output: URL
  public let inputFrames: Int
  public let outputFrames: Int
  public let sceneResets: Int
  public let elapsedSeconds: Double
  public let motionBackend: String
  public var timing: MediaStageTiming? = nil
}

/// Wall time spent awaiting each stage, including GPU completion where required.
/// Setup and cold kernel compilation remain visible in end-to-end latency.
/// Audio runs concurrently, so stage times are not an additive wall-time total.
public struct MediaStageTiming: Codable, Sendable {
  public var setupSeconds: Double = 0
  public var decodingSeconds: Double = 0
  public var motionSeconds: Double = 0
  public var renderingSeconds: Double = 0
  public var generationSeconds: Double = 0
  public var superResolutionSeconds: Double? = nil
  public var encodingSeconds: Double = 0
  public var audioSeconds: Double = 0
}
