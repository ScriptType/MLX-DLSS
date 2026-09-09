import Cmlx
import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import Metal
import MLX

/// Source interpretation. Working images are always linear BT.2020 RGB in cd/m²
/// (nits); reference white is a normalization policy, never an HDR clipping limit.
public struct MLXHDRColorMetadata: Equatable, Sendable {
  public enum Transfer: UInt32, Sendable { case bt1886, sRGB, pq, hlg, linear }
  public enum Primaries: UInt32, Sendable { case bt709, bt2020, displayP3 }
  public enum Matrix: UInt32, Sendable { case bt709, bt2020, bt601 }
  public enum ChromaLocation: UInt32, Sendable { case center, left, topLeft, top, bottomLeft, bottom }
  public var transfer: Transfer
  public var primaries: Primaries
  public var matrix: Matrix
  public var fullRange: Bool
  public var chromaLocation: ChromaLocation
  public var referenceWhiteNits: Float
  public var hlgPeakNits: Float
  public var sourceTags: [String: String]
  public var masteringDisplay: Data?
  public var contentLightLevel: Data?
  public var assumptions: [String]

  public init(transfer: Transfer = .bt1886, primaries: Primaries = .bt709,
    matrix: Matrix = .bt709, fullRange: Bool = false, chromaLocation: ChromaLocation = .left,
    referenceWhiteNits: Float = 203, hlgPeakNits: Float = 1000,
    sourceTags: [String: String] = [:], masteringDisplay: Data? = nil,
    contentLightLevel: Data? = nil, assumptions: [String] = []) {
    self.transfer = transfer; self.primaries = primaries; self.matrix = matrix
    self.fullRange = fullRange; self.chromaLocation = chromaLocation
    self.referenceWhiteNits = referenceWhiteNits; self.hlgPeakNits = hlgPeakNits
    self.sourceTags = sourceTags; self.masteringDisplay = masteringDisplay
    self.contentLightLevel = contentLightLevel; self.assumptions = assumptions
  }

  public static func read(from buffer: CVPixelBuffer, referenceWhiteNits: Float = 203,
    hlgPeakNits: Float = 1000, fallbackTags: [String: String] = [:]) throws -> Self {
    func attachment(_ key: CFString) -> CFTypeRef? { CVBufferCopyAttachment(buffer, key, nil) }
    func tag(_ key: CFString) -> String? { attachment(key).map { String(describing: $0) } ?? fallbackTags[key as String] }
    var result = Self(referenceWhiteNits: referenceWhiteNits, hlgPeakNits: hlgPeakNits)
    let format = CVPixelBufferGetPixelFormatType(buffer)
    result.fullRange = [kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
      kCVPixelFormatType_420YpCbCr10BiPlanarFullRange].contains(format)
    if let fields = attachment(kCVImageBufferFieldCountKey) as? NSNumber, fields.intValue > 1 {
      throw MLXMediaError("Interlaced HDR import requires a deinterlacing policy")
    }
    let transfer = tag(kCVImageBufferTransferFunctionKey)
    switch transfer {
    case String(describing: kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ): result.transfer = .pq
    case String(describing: kCVImageBufferTransferFunction_ITU_R_2100_HLG): result.transfer = .hlg
    case String(describing: kCVImageBufferTransferFunction_sRGB): result.transfer = .sRGB
    case String(describing: kCVImageBufferTransferFunction_Linear): result.transfer = .linear
    case String(describing: kCVImageBufferTransferFunction_ITU_R_709_2), String(describing: kCVImageBufferTransferFunction_ITU_R_2020): result.transfer = .bt1886
    case nil: result.assumptions.append("Missing transfer: BT.1886 ideal-black display EOTF")
    default: throw MLXMediaError("Unsupported source transfer: \(transfer!)")
    }
    let primaries = tag(kCVImageBufferColorPrimariesKey)
    switch primaries {
    case String(describing: kCVImageBufferColorPrimaries_ITU_R_2020): result.primaries = .bt2020
    case String(describing: kCVImageBufferColorPrimaries_P3_D65): result.primaries = .displayP3
    case String(describing: kCVImageBufferColorPrimaries_ITU_R_709_2): break
    case nil: result.assumptions.append("Missing primaries: BT.709")
    default: throw MLXMediaError("Unsupported source primaries: \(primaries!)")
    }
    let matrix = tag(kCVImageBufferYCbCrMatrixKey)
    switch matrix {
    case String(describing: kCVImageBufferYCbCrMatrix_ITU_R_2020): result.matrix = .bt2020
    case String(describing: kCVImageBufferYCbCrMatrix_ITU_R_601_4): result.matrix = .bt601
    case String(describing: kCVImageBufferYCbCrMatrix_ITU_R_709_2): break
    case nil: result.assumptions.append("Missing YCbCr matrix: BT.709")
    default: throw MLXMediaError("Unsupported YCbCr matrix: \(matrix!)")
    }
    let chroma = tag(kCVImageBufferChromaLocationTopFieldKey)
    switch chroma {
    case String(describing: kCVImageBufferChromaLocation_Center): result.chromaLocation = .center
    case String(describing: kCVImageBufferChromaLocation_Left): result.chromaLocation = .left
    case String(describing: kCVImageBufferChromaLocation_TopLeft): result.chromaLocation = .topLeft
    case String(describing: kCVImageBufferChromaLocation_Top): result.chromaLocation = .top
    case String(describing: kCVImageBufferChromaLocation_BottomLeft): result.chromaLocation = .bottomLeft
    case String(describing: kCVImageBufferChromaLocation_Bottom): result.chromaLocation = .bottom
    case nil: result.assumptions.append("Missing chroma location: progressive left")
    default: throw MLXMediaError("Unsupported chroma location: \(chroma!)")
    }
    for key in [kCVImageBufferTransferFunctionKey, kCVImageBufferColorPrimariesKey,
      kCVImageBufferYCbCrMatrixKey, kCVImageBufferChromaLocationTopFieldKey] {
      if let value = tag(key) { result.sourceTags[key as String] = value }
    }
    result.masteringDisplay = attachment(kCVImageBufferMasteringDisplayColorVolumeKey) as? Data
    result.contentLightLevel = attachment(kCVImageBufferContentLightLevelInfoKey) as? Data
    return result
  }
}

public struct MLXHDRFrameMetadata: Sendable {
  public var time: CMTime
  public var duration: CMTime
  public var sourceID: String
  public var streamID: UInt64
  public var frameIndex: UInt64
  public var generation: UInt64
  public var crop: CGRect
  public var transform: CGAffineTransform
  public var pixelAspectRatio: Double
  public var color: MLXHDRColorMetadata

  public init(time: CMTime, duration: CMTime, sourceID: String, streamID: UInt64 = 1,
    frameIndex: UInt64, generation: UInt64 = 0, crop: CGRect = .zero,
    transform: CGAffineTransform = .identity, pixelAspectRatio: Double = 1,
    color: MLXHDRColorMetadata = .init()) {
    self.time = time; self.duration = duration; self.sourceID = sourceID
    self.streamID = streamID; self.frameIndex = frameIndex; self.generation = generation
    self.crop = crop; self.transform = transform; self.pixelAspectRatio = pixelAspectRatio; self.color = color
  }
}

/// Immutable completed original. No proxy conversion or quantization occurs at
/// this boundary. Retaining the frame retains the decoder and GPU storage owners.
public final class MLXHDRFrame: Sendable {
  public let original: MLXVideoFrame
  public let metadata: MLXHDRFrameMetadata
  public let sourcePixelBuffer: MLXPixelBuffer?
  public let importGPUSeconds: Double?
  public init(original: MLXVideoFrame, metadata: MLXHDRFrameMetadata, sourcePixelBuffer: MLXPixelBuffer? = nil,
    importGPUSeconds: Double? = nil) {
    self.original = original; self.metadata = metadata; self.sourcePixelBuffer = sourcePixelBuffer
    self.importGPUSeconds = importGPUSeconds
  }
}

/// Session-scoped NV12/P010 conversion. Textures use hardware bilinear chroma
/// reconstruction with the declared siting. The returned RGB buffer is complete;
/// no CPU pixel readback is performed and Metal owners survive command completion.
public actor MLXHDRImporter {
  private let device: any MTLDevice
  private let queue: any MTLCommandQueue
  private let pipeline: any MTLComputePipelineState
  private let cache: CVMetalTextureCache

  public init() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw MLXMediaError("Metal is unavailable")
    }
    self.device = device; self.queue = queue
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try device.makeLibrary(source: Self.shader, options: options)
    pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "importHDR")!)
    var cache: CVMetalTextureCache?
    guard CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess, let cache else {
      throw MLXMediaError("Cannot create planar texture cache")
    }
    self.cache = cache
  }

  public func importFrame(pixelBuffer: MLXPixelBuffer, metadata: MLXHDRFrameMetadata) async throws -> MLXHDRFrame {
    let source = pixelBuffer.buffer, width = pixelBuffer.width, height = pixelBuffer.height
    let format = CVPixelBufferGetPixelFormatType(source)
    let tenBit = [kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr10BiPlanarFullRange].contains(format)
    guard tenBit || [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange].contains(format),
      CVPixelBufferGetPlaneCount(source) == 2, width > 0, height > 0, width <= Int(Int32.max) / height / 3 else {
      throw MLXMediaError("HDR import requires NV12 or P010 decoder planes")
    }
    guard metadata.color.referenceWhiteNits.isFinite, metadata.color.referenceWhiteNits > 0,
      metadata.color.hlgPeakNits.isFinite, (400...2000).contains(metadata.color.hlgPeakNits),
      metadata.pixelAspectRatio.isFinite, metadata.pixelAspectRatio > 0 else { throw MLXMediaError("Invalid HDR luminance or geometry policy") }
    let isFullRange = [kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVPixelFormatType_420YpCbCr10BiPlanarFullRange].contains(format)
    guard metadata.color.fullRange == isFullRange else { throw MLXMediaError("HDR range metadata disagrees with decoder storage") }
    func plane(_ index: Int, _ format: MTLPixelFormat) throws -> CVMetalTexture {
      var result: CVMetalTexture?
      let status = CVMetalTextureCacheCreateTextureFromImage(nil, cache, source, nil, format,
        CVPixelBufferGetWidthOfPlane(source, index), CVPixelBufferGetHeightOfPlane(source, index), index, &result)
      guard status == kCVReturnSuccess, let result else { throw MLXMediaError("Cannot bind decoder plane \(index): \(status)") }
      return result
    }
    let y = try plane(0, tenBit ? .r16Unorm : .r8Unorm)
    let uv = try plane(1, tenBit ? .rg16Unorm : .rg8Unorm)
    let count = width * height * 3
    guard let output = device.makeBuffer(length: (count * 4 + 4095) & ~4095, options: .storageModeShared),
      let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
      throw MLXMediaError("Cannot allocate HDR conversion resources")
    }
    var parameters: [UInt32] = [UInt32(width), UInt32(height), tenBit ? 1 : 0, isFullRange ? 1 : 0,
      metadata.color.transfer.rawValue, metadata.color.primaries.rawValue, metadata.color.matrix.rawValue,
      metadata.color.chromaLocation.rawValue, metadata.color.referenceWhiteNits.bitPattern, metadata.color.hlgPeakNits.bitPattern]
    encoder.setComputePipelineState(pipeline)
    encoder.setTexture(CVMetalTextureGetTexture(y), index: 0)
    encoder.setTexture(CVMetalTextureGetTexture(uv), index: 1)
    encoder.setBuffer(output, offset: 0, index: 0)
    encoder.setBytes(&parameters, length: parameters.count * 4, index: 1)
    encoder.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    encoder.endEncoding()
    let owners = ImportOwners(source: pixelBuffer, y: y, uv: uv, output: output)
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
      command.addCompletedHandler { [owners] completed in
        withExtendedLifetime(owners) {
          if let error = completed.error { continuation.resume(throwing: error) } else { continuation.resume() }
        }
      }
      command.commit()
    }
    var shape = [Int32(1), Int32(height), Int32(width), Int32(3)]
    let owner = Unmanaged.passRetained(owners).toOpaque()
    let array = MLXArray(mlx_array_new_data_managed_payload(output.contents(), &shape, 4, MLX_FLOAT32, owner) { payload in
      if let payload { Unmanaged<ImportOwners>.fromOpaque(payload).release() }
    })
    let duration = command.gpuEndTime - command.gpuStartTime
    return MLXHDRFrame(original: MLXVideoFrame(array), metadata: metadata, sourcePixelBuffer: pixelBuffer,
      importGPUSeconds: duration > 0 ? duration : nil)
  }

  private final class ImportOwners: @unchecked Sendable {
    let source: MLXPixelBuffer; let y: CVMetalTexture; let uv: CVMetalTexture; let output: any MTLBuffer
    init(source: MLXPixelBuffer, y: CVMetalTexture, uv: CVMetalTexture, output: any MTLBuffer) {
      self.source = source; self.y = y; self.uv = uv; self.output = output
    }
  }

  private static let shader = #"""
    #include <metal_stdlib>
    using namespace metal;
    float pq(float x) {
      float p = pow(max(x, 0.0f), 1.0f / 78.84375f);
      return 10000.0f * pow(max(p - 0.8359375f, 0.0f) / max(18.8515625f - 18.6875f * p, 1e-6f), 1.0f / 0.1593017578125f);
    }
    float hlg(float x) { return x <= 0.5f ? x*x/3.0f : (exp((x-0.55991073f)/0.17883277f)+0.28466892f)/12.0f; }
    float transfer(float x, uint t, float white) {
      x = max(x, 0.0f);
      if (t == 2) return pq(x);
      if (t == 3) return hlg(x);
      if (t == 4) return x * white;
      if (t == 1) return (x <= 0.04045f ? x / 12.92f : pow((x + 0.055f)/1.055f, 2.4f)) * white;
      return pow(x, 2.4f) * white;
    }
    kernel void importHDR(texture2d<float, access::read> yPlane [[texture(0)]],
      texture2d<float, access::sample> uvPlane [[texture(1)]], device float *output [[buffer(0)]],
      constant uint *p [[buffer(1)]], uint2 pos [[thread_position_in_grid]]) {
      if (pos.x >= p[0] || pos.y >= p[1]) return;
      constexpr sampler chroma(coord::pixel, address::clamp_to_edge, filter::linear);
      // Chroma sample centres expressed in luma sample-index coordinates.
      float2 origin = float2(0.5f, 0.5f);
      if (p[7] == 1 || p[7] == 2 || p[7] == 4) origin.x = 0.0f;
      if (p[7] == 2 || p[7] == 3) origin.y = 0.0f;
      if (p[7] == 4 || p[7] == 5) origin.y = 1.0f;
      float2 uv = uvPlane.sample(chroma, (float2(pos) - origin) * 0.5f + 0.5f).rg;
      float y = yPlane.read(pos).r;
      float codes = p[2] ? 65535.0f / 64.0f : 255.0f;
      y *= codes; uv *= codes;
      float scale = p[2] ? 4.0f : 1.0f;
      y = p[3] ? y / (p[2] ? 1023.0f : 255.0f) : (y - 16.0f * scale) / (219.0f * scale);
      uv = (uv - 128.0f * scale) / (p[3] ? (p[2] ? 1023.0f : 255.0f) : 224.0f * scale);
      float kr = p[6] == 1 ? 0.2627f : p[6] == 2 ? 0.299f : 0.2126f;
      float kb = p[6] == 1 ? 0.0593f : p[6] == 2 ? 0.114f : 0.0722f;
      float r = y + 2.0f*(1.0f-kr)*uv.y;
      float b = y + 2.0f*(1.0f-kb)*uv.x;
      float g = (y-kr*r-kb*b)/(1.0f-kr-kb);
      float white = as_type<float>(p[8]);
      float3 linear = float3(transfer(r,p[4],white), transfer(g,p[4],white), transfer(b,p[4],white));
      if (p[4] == 3) {
        float peak = as_type<float>(p[9]);
        float gamma = 1.2f + 0.42f * log10(peak/1000.0f);
        float lum = dot(linear, p[5] == 1 ? float3(0.2627f,0.6780f,0.0593f) : p[5] == 2 ? float3(0.2289746f,0.6917385f,0.0792869f) : float3(0.2126f,0.7152f,0.0722f));
        linear *= peak * pow(max(lum, 0.0f), gamma - 1.0f);
      }
      if (p[5] == 0) linear = float3(
        dot(linear,float3(0.6274039f,0.3292830f,0.0433131f)),
        dot(linear,float3(0.0690973f,0.9195404f,0.0113623f)),
        dot(linear,float3(0.0163914f,0.0880133f,0.8955953f)));
      if (p[5] == 2) linear = float3(
        dot(linear,float3(0.7538330f,0.1985974f,0.0475696f)),
        dot(linear,float3(0.0457438f,0.9417772f,0.0124790f)),
        dot(linear,float3(-0.0012103f,0.0176017f,0.9836086f)));
      uint i = (pos.y*p[0]+pos.x)*3;
      output[i]=linear.r; output[i+1]=linear.g; output[i+2]=linear.b;
    }
    """#
}
