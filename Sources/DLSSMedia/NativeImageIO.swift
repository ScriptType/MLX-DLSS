import CoreImage
import CoreVideo
import DLSSMLX
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers

/// ImageIO and Core Image share the same sRGB, top-to-bottom RGB contract as
/// decoded video. The IOSurface owner survives until the MLX import completes.
public actor NativeImageIO {
  private let context: CIContext
  private let colourSpace: CGColorSpace
  private var previewWriter: (width: Int, height: Int, writer: MLXPixelBufferWriter)?

  public init() throws {
    guard let device = MTLCreateSystemDefaultDevice(),
      let colourSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
      throw MLXMediaError("Metal and sRGB are required for native media processing")
    }
    self.colourSpace = colourSpace
    context = CIContext(mtlDevice: device, options: [.cacheIntermediates: false])
  }

  public func read(_ url: URL) throws -> MLXVideoFrame {
    guard let image = CIImage(contentsOf: url, options: [.applyOrientationProperty: true]) else {
      throw MLXMediaError("Cannot decode image: \(url.lastPathComponent)")
    }
    return try MLXVideoFrame(pixelBuffer: convert(image))
  }

  func convert(_ buffer: MLXPixelBuffer, transform: CGAffineTransform = .identity) throws -> MLXPixelBuffer {
    try convert(CIImage(cvPixelBuffer: buffer.buffer, options: [.colorSpace: colourSpace]).transformed(by: transform))
  }

  func resize(_ buffer: MLXPixelBuffer, width: Int, height: Int) throws -> MLXPixelBuffer {
    let scale = CGAffineTransform(scaleX: Double(width) / Double(buffer.width),
      y: Double(height) / Double(buffer.height))
    return try convert(CIImage(cvPixelBuffer: buffer.buffer, options: [.colorSpace: colourSpace]).transformed(by: scale))
  }

  private func convert(_ image: CIImage) throws -> MLXPixelBuffer {
    let extent = image.extent.integral
    guard !extent.isInfinite, !extent.isEmpty, extent.width <= 16384, extent.height <= 16384 else {
      throw MLXMediaError("Unsupported image extent")
    }
    let buffer = try NativePixelBuffers.make(width: Int(extent.width), height: Int(extent.height),
      format: kCVPixelFormatType_64RGBAHalf)
    let normalized = image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
    context.render(normalized, to: buffer.buffer, bounds: CGRect(x: 0, y: 0,
      width: extent.width, height: extent.height), colorSpace: colourSpace)
    return buffer
  }

  public func write(_ frame: MLXVideoFrame, to url: URL) async throws {
    let type: UTType
    switch url.pathExtension.lowercased() {
    case "png": type = .png
    case "jpg", "jpeg": type = .jpeg
    case "tif", "tiff": type = .tiff
    case "heic", "heif": type = .heic
    default: throw MLXMediaError("Image output must be PNG, JPEG, TIFF or HEIC")
    }
    guard !FileManager.default.fileExists(atPath: url.path) else {
      throw MLXMediaError("Output already exists: \(url.lastPathComponent)")
    }
    let writer = try MLXPixelBufferWriter(width: frame.width, height: frame.height, halfOutput: true)
    let buffer = try await writer.write(frame)
    let image = CIImage(cvPixelBuffer: buffer.buffer, options: [.colorSpace: colourSpace])
    let format: CIFormat = type == .png || type == .tiff ? .RGBA16 : .RGBA8
    let staging = url.deletingLastPathComponent().appendingPathComponent(".mlxdlss-image-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false,
      attributes: [.posixPermissions: 0o700])
    defer { try? FileManager.default.removeItem(at: staging) }
    let temporary = staging.appendingPathComponent(url.lastPathComponent)
    guard let cgImage = context.createCGImage(image, from: image.extent, format: format, colorSpace: colourSpace),
      let destination = CGImageDestinationCreateWithURL(temporary as CFURL, type.identifier as CFString, 1, nil) else {
      throw MLXMediaError("Cannot create image output")
    }
    CGImageDestinationAddImage(destination, cgImage, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
    guard CGImageDestinationFinalize(destination) else {
      throw MLXMediaError("Cannot finish image output")
    }
    try Task.checkCancellation()
    try FileManager.default.moveItem(at: temporary, to: url)
  }

  /// An in-memory display image; no encoded file or float32 CPU copy is needed.
  public func displayImage(_ frame: MLXVideoFrame) async throws -> CGImage {
    if previewWriter?.width != frame.width || previewWriter?.height != frame.height {
      previewWriter = (frame.width, frame.height,
        try MLXPixelBufferWriter(width: frame.width, height: frame.height, halfOutput: true))
    }
    let buffer = try await previewWriter!.writer.write(frame)
    let image = CIImage(cvPixelBuffer: buffer.buffer, options: [.colorSpace: colourSpace])
    guard let result = context.createCGImage(image, from: image.extent, format: .RGBA8,
      colorSpace: colourSpace, deferred: false) else { throw MLXMediaError("Cannot display the preview") }
    return result
  }
}

enum NativePixelBuffers {
  static func make(width: Int, height: Int, format: OSType) throws -> MLXPixelBuffer {
    guard width > 0, height > 0, width <= 16384, height <= 16384 else {
      throw MLXMediaError("Invalid native pixel buffer extent")
    }
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, [
      kCVPixelBufferIOSurfacePropertiesKey: [:], kCVPixelBufferMetalCompatibilityKey: true,
    ] as CFDictionary, &buffer)
    guard status == kCVReturnSuccess, let buffer else {
      throw MLXMediaError("Cannot allocate native pixel buffer: \(status)")
    }
    return MLXPixelBuffer(buffer)
  }
}
