import DLSSMedia
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// One JSON request/response per line; the process owns one cached preview session.
enum PreviewStreamCommand {
  private struct Request: Decodable {
    let input: String
    let video: Bool
    let time: Double
    let options: [String]
  }

  static func run(arguments: [String]) async throws {
    guard arguments.isEmpty else { throw CLIError.usage("preview-stream takes JSON lines on stdin") }
    guard #available(macOS 26.0, *) else { throw CLIError.usage("native preview requires macOS 26 or newer") }
    let session = try NativeMediaPreview()
    while let line = readLine() {
      do {
        let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8))
        let parsed = try ProcessMediaCommand.parse(
          arguments: [request.input, "--output", request.input] + request.options,
          video: request.video, requireEffect: false)
        let result = try await session.render(.init(input: parsed.input, isVideo: request.video,
          time: request.time, options: parsed.options))
        try CLIOutput.writeJSON([
          "original": try png(result.original), "processed": try png(result.processed),
          "width": result.processed.width, "height": result.processed.height,
          "time": result.time, "duration": result.duration, "frameInterval": result.frameInterval,
          "historyFrames": result.historyFrames, "elapsedSeconds": result.elapsedSeconds,
        ])
      } catch {
        try CLIOutput.writeJSON(["error": String(describing: error)])
      }
    }
  }

  private static func png(_ image: CGImage) throws -> String {
    let data = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
      throw CLIError.usage("cannot create preview image")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw CLIError.usage("cannot encode preview image") }
    return (data as Data).base64EncodedString()
  }
}
