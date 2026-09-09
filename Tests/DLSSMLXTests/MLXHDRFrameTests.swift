import CoreMedia
import CoreVideo
import Foundation
import XCTest
@testable import DLSSMLX

final class MLXHDRFrameTests: XCTestCase, @unchecked Sendable {
  func testPlanarSDRPQAndHLGGreyAgainstIndependentTransferReference() async throws {
    let importer = try MLXHDRImporter()
    for tenBit in [false, true] {
      for fullRange in [false, true] {
        for transfer in [MLXHDRColorMetadata.Transfer.bt1886, .sRGB, .pq, .hlg] {
          let scale: Double = tenBit ? 4 : 1
          let maxCode: Double = tenBit ? 1023 : 255
          for signal in [0.0, 0.05, 0.5, 0.75, 1.0] {
            let code = Int((fullRange ? signal * maxCode : (16 + 219 * signal) * scale).rounded())
            let buffer = try fixture(tenBit: tenBit, fullRange: fullRange, y: code, cb: Int(128*scale), cr: Int(128*scale))
            let color = MLXHDRColorMetadata(transfer: transfer, primaries: .bt2020, matrix: .bt2020, fullRange: fullRange)
            let metadata = MLXHDRFrameMetadata(time: CMTime(value: 1001, timescale: 24000),
              duration: CMTime(value: 1001, timescale: 24000), sourceID: "numeric-grey", frameIndex: 1,
              generation: 7, crop: CGRect(x: 1, y: 0, width: 4, height: 4), pixelAspectRatio: 4.0/3, color: color)
            let frame = try await importer.importFrame(pixelBuffer: buffer, metadata: metadata)
            let e = fullRange ? Double(code)/maxCode : (Double(code)/scale - 16)/219
            let expected = decode(e, transfer: transfer)
            for value in values(frame.original) {
              XCTAssertEqual(Double(value), expected, accuracy: max(0.003, expected*0.0002), "\(transfer), \(tenBit), \(fullRange), \(signal)")
            }
            XCTAssertTrue(frame.sourcePixelBuffer!.buffer === buffer.buffer)
            XCTAssertEqual(frame.metadata.time.value, 1001)
            XCTAssertEqual(frame.metadata.time.timescale, 24000)
            XCTAssertEqual(frame.metadata.generation, 7)
            XCTAssertEqual(frame.metadata.crop.width, 4)
            XCTAssertEqual(frame.metadata.pixelAspectRatio, 4.0/3)
          }
        }
      }
    }
  }

  func testPQRedPreservesSaturationAndFloatOutputAboveReferenceWhite() async throws {
    let r = encodePQ(1000), g = encodePQ(20), b = encodePQ(2)
    let y = 0.2627*r + 0.678*g + 0.0593*b
    let cb = (b-y)/(2*(1-0.0593)), cr = (r-y)/(2*(1-0.2627))
    let yc = Int((64+876*y).rounded()), cbc = Int((512+896*cb).rounded()), crc = Int((512+896*cr).rounded())
    let buffer = try fixture(tenBit: true, fullRange: false, y: yc, cb: cbc, cr: crc)
    let importer = try MLXHDRImporter()
    let color = MLXHDRColorMetadata(transfer: .pq, primaries: .bt2020, matrix: .bt2020)
    let frame = try await importer.importFrame(pixelBuffer: buffer, metadata: .init(time: .zero,
      duration: CMTime(value: 1, timescale: 24), sourceID: "red", frameIndex: 0, color: color))
    let rgb = values(frame.original)
    XCTAssertEqual(rgb[0], 1000, accuracy: 8)
    XCTAssertEqual(rgb[1], 20, accuracy: 0.5)
    XCTAssertEqual(rgb[2], 2, accuracy: 0.1)
    let writer = try MLXPixelBufferWriter(width: 6, height: 4, halfOutput: true)
    let packed = try await writer.write(frame.original)
    let unpacked = try MLXVideoFrame(pixelBuffer: packed)
    XCTAssertGreaterThan(values(unpacked)[0], 4 * color.referenceWhiteNits)
  }

  func testMetadataReadsSourceAttachmentsAndRejectsUnknownTransfer() throws {
    let buffer = try fixture(tenBit: true, fullRange: false, y: 64, cb: 512, cr: 512).buffer
    CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_2020, .shouldPropagate)
    CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_2020, .shouldPropagate)
    let mastering = Data(repeating: 1, count: 24)
    CVBufferSetAttachment(buffer, kCVImageBufferMasteringDisplayColorVolumeKey, mastering as CFData, .shouldPropagate)
    let color = try MLXHDRColorMetadata.read(from: buffer)
    XCTAssertEqual(color.transfer, .pq)
    XCTAssertEqual(color.primaries, .bt2020)
    XCTAssertEqual(color.masteringDisplay, mastering)
    CVBufferRemoveAttachment(buffer, kCVImageBufferTransferFunctionKey)
    XCTAssertThrowsError(try MLXHDRColorMetadata.read(from: buffer, fallbackTags: [kCVImageBufferTransferFunctionKey as String: "unknown"]))
  }

  func testSDRPrimaryConversionUsesBT2020WorkingSpace() async throws {
    let buffer = try fixture(tenBit: false, fullRange: false, y: 63, cb: 102, cr: 240)
    let importer = try MLXHDRImporter()
    let frame = try await importer.importFrame(pixelBuffer: buffer, metadata: .init(time: .zero,
      duration: CMTime(value: 1, timescale: 24), sourceID: "sdr-red", frameIndex: 0))
    let v = values(frame.original)
    // BT.709 red transformed to BT.2020 has positive G/B working components.
    XCTAssertGreaterThan(v[0], 120)
    XCTAssertGreaterThan(v[1], 12)
    XCTAssertGreaterThan(v[2], 2)
    XCTAssertLessThan(v[1], 17)
  }

  func testBilinearChromaReconstructionRespectsLeftAndCenterSiting() async throws {
    let buffer = try fixture(tenBit: true, fullRange: false, y: 500, cb: 512, cr: 512)
    CVPixelBufferLockBaseAddress(buffer.buffer, [])
    let base = CVPixelBufferGetBaseAddressOfPlane(buffer.buffer, 1)!
    for row in 0..<2 {
      for (x, code) in [256, 512, 768].enumerated() {
        base.storeBytes(of: UInt16(code << 6),
          toByteOffset: row * CVPixelBufferGetBytesPerRowOfPlane(buffer.buffer, 1) + x*4, as: UInt16.self)
      }
    }
    CVPixelBufferUnlockBaseAddress(buffer.buffer, [])
    let importer = try MLXHDRImporter()
    for (siting, reconstructed) in [(MLXHDRColorMetadata.ChromaLocation.left, [256,384,512,640,768,768]),
      (.center, [256,320,448,576,704,768])] {
      let color = MLXHDRColorMetadata(transfer: .linear, primaries: .bt2020, matrix: .bt2020, chromaLocation: siting)
      let frame = try await importer.importFrame(pixelBuffer: buffer, metadata: .init(time: .zero,
        duration: CMTime(value: 1, timescale: 24), sourceID: "chroma", frameIndex: 0, color: color))
      let rgb = values(frame.original)
      for x in 0..<6 {
        let cb = Float(reconstructed[x]-512)/896
        let expectedBlue = max(0, Float(500-64)/876 + 2*(1-0.0593)*cb)*203
        XCTAssertEqual(rgb[x*3+2], expectedBlue, accuracy: 0.0001)
      }
    }
  }

  private func values(_ frame: MLXVideoFrame) -> [Float] {
    frame.copyRGBData().withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  }
  private func encodePQ(_ nits: Double) -> Double {
    let p = pow(nits/10000, 2610.0/16384)
    return pow((3424.0/4096 + 2413.0/128*p)/(1+2392.0/128*p), 2523.0/32)
  }
  private func decode(_ e: Double, transfer: MLXHDRColorMetadata.Transfer) -> Double {
    switch transfer {
    case .bt1886: return 203 * pow(max(e,0),2.4)
    case .sRGB: return 203 * (e <= 0.04045 ? e/12.92 : pow((e+0.055)/1.055,2.4))
    case .pq:
      let p = pow(max(e,0),32.0/2523)
      return 10000 * pow(max(p-3424.0/4096,0)/(2413.0/128-2392.0/128*p),16384.0/2610)
    case .hlg:
      let a = 0.17883277, b = 1-4*0.17883277, c = 0.5-0.17883277*log(4*0.17883277)
      let scene = e <= 0.5 ? e*e/3 : (exp((e-c)/a)+b)/12
      return 1000 * pow(scene,1.2)
    case .linear: return e*203
    }
  }
  private func fixture(tenBit: Bool, fullRange: Bool, y: Int, cb: Int, cr: Int) throws -> MLXPixelBuffer {
    let format = tenBit
      ? (fullRange ? kCVPixelFormatType_420YpCbCr10BiPlanarFullRange : kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange)
      : (fullRange ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
    var result: CVPixelBuffer?
    XCTAssertEqual(CVPixelBufferCreate(nil, 6, 4, format, [kCVPixelBufferIOSurfacePropertiesKey: [:],
      kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &result), kCVReturnSuccess)
    let buffer = try XCTUnwrap(result)
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    for plane in 0..<2 {
      let base = CVPixelBufferGetBaseAddressOfPlane(buffer, plane)!
      for row in 0..<CVPixelBufferGetHeightOfPlane(buffer, plane) {
        for x in 0..<6 {
          let value = plane == 0 ? y : x%2 == 0 ? cb : cr
          let offset = row * CVPixelBufferGetBytesPerRowOfPlane(buffer, plane) + x*(tenBit ? 2 : 1)
          if tenBit { base.storeBytes(of: UInt16(value << 6), toByteOffset: offset, as: UInt16.self) }
          else { base.storeBytes(of: UInt8(value), toByteOffset: offset, as: UInt8.self) }
        }
      }
    }
    return MLXPixelBuffer(buffer)
  }
}
