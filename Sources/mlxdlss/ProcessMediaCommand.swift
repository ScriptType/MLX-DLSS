import DLSSCore
import DLSSMedia
import DLSSMLX
import Foundation

enum ProcessMediaCommand {
  struct Parsed {
    let input: URL
    let output: URL
    let options: MediaProcessingOptions
  }

  static func run(arguments: [String], video: Bool) async throws {
    let parsed = try parse(arguments: arguments, video: video)
    let processor = NativeMediaProcessor()
    let result: MediaProcessingResult
    if video {
      guard #available(macOS 26.0, *) else {
        throw CLIError.usage("native video processing requires macOS 26 or newer")
      }
      result = try await processor.processVideo(input: parsed.input, output: parsed.output, options: parsed.options) { progress in
        if progress.inputFrames == 1 || progress.inputFrames % 10 == 0 {
          let line = "\(progress.inputFrames)/\(progress.estimatedInputFrames) input frames, \(progress.outputFrames) output, \(String(format: "%.1f", progress.elapsedSeconds)) s\n"
          FileHandle.standardError.write(Data(line.utf8))
        }
      }
    } else {
      result = try await processor.processImage(input: parsed.input, output: parsed.output, options: parsed.options)
    }
    try CLIOutput.writeEncodable(result)
  }

  static func parse(arguments: [String], video: Bool, requireEffect: Bool = true) throws -> Parsed {
    guard let input = arguments.first, !input.hasPrefix("--") else {
      throw CLIError.usage("process-\(video ? "video" : "image") requires INPUT --output PATH and --model, --vsr-weights or --framegen-weights")
    }
    let common = ["--output", "--model", "--profile", "--precision", "--intensity",
      "--processing-scale", "--detail-strength", "--colour-strength", "--detail-radius", "--vsr-weights"]
    let videoOptions = ["--framegen-weights", "--order", "--factor", "--temporal", "--motion",
      "--scene-cut-threshold", "--slow-motion", "--audio", "--codec", "--bitrate", "--start-frame", "--frames"]
    let known = common + (video ? videoOptions : [])
    var values: [String: String] = [:]
    var index = 1
    while index < arguments.count {
      let key = arguments[index]
      guard known.contains(key), values[key] == nil, index + 1 < arguments.count else {
        throw CLIError.usage("unknown, duplicate or incomplete media option '\(key)'")
      }
      values[key] = arguments[index + 1]
      index += 2
    }
    guard let output = values["--output"] else { throw CLIError.usage("native media processing requires --output PATH") }
    var options = MediaProcessingOptions(renderingModel: values["--model"].map { URL(fileURLWithPath: $0) },
      frameGenerationWeights: values["--framegen-weights"].map { URL(fileURLWithPath: $0) },
      superResolutionWeights: values["--vsr-weights"].map { URL(fileURLWithPath: $0) })
    func float(_ key: String, _ fallback: Float) throws -> Float {
      guard let text = values[key] else { return fallback }
      guard let result = Float(text), result.isFinite else { throw CLIError.usage("\(key) requires a finite number") }
      return result
    }
    func integer(_ key: String, _ fallback: Int) throws -> Int {
      guard let text = values[key] else { return fallback }
      guard let result = Int(text) else { throw CLIError.usage("\(key) requires an integer") }
      return result
    }
    func boolean(_ key: String, _ fallback: Bool) throws -> Bool {
      guard let text = values[key] else { return fallback }
      guard ["on", "off"].contains(text) else { throw CLIError.usage("\(key) must be on or off") }
      return text == "on"
    }
    func choice<T: RawRepresentable>(_ key: String, _ fallback: T) throws -> T where T.RawValue == String {
      guard let text = values[key] else { return fallback }
      guard let result = T(rawValue: text) else { throw CLIError.usage("invalid \(key) value '\(text)'") }
      return result
    }
    options.profile = try choice("--profile", video ? .standard : .natural)
    options.precision = try choice("--precision", .float16)
    options.order = try choice("--order", .renderingThenGeneration)
    options.motion = try choice("--motion", .automatic)
    options.codec = try choice("--codec", .h264)
    options.processingScale = try float("--processing-scale", 1)
    options.detailStrength = try float("--detail-strength", 1)
    options.colourStrength = try float("--colour-strength", 1)
    options.detailRadius = try float("--detail-radius", 4)
    options.intensity = try float("--intensity", 1)
    options.sceneCutThreshold = try float("--scene-cut-threshold", 0.3)
    options.temporal = try boolean("--temporal", true)
    options.slowMotion = try boolean("--slow-motion", false)
    options.includeAudio = try boolean("--audio", true)
    options.frameGenerationFactor = try integer("--factor", 2)
    options.startFrame = try integer("--start-frame", 0)
    if values["--frames"] != nil { options.frameLimit = try integer("--frames", 0) }
    if values["--bitrate"] != nil { options.bitrate = try integer("--bitrate", 0) }
    if requireEffect || options.renderingModel != nil || options.frameGenerationWeights != nil
      || options.superResolutionWeights != nil {
      try options.validate()
    }
    return Parsed(input: URL(fileURLWithPath: input), output: URL(fileURLWithPath: output), options: options)
  }
}
