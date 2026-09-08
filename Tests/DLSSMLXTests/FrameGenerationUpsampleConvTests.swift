import Foundation
import MLX
import XCTest
@testable import DLSSMLX

final class FrameGenerationUpsampleConvTests: XCTestCase {
  func testUpsampleConvolutionMatchesSeparateRoundingAtEdges() {
    for (height, width) in [(1, 1), (3, 5), (7, 9)] {
      let input = (MLXRandom.normal([2, width, height, 16]) * 0.3).asType(.float16).transposed(0, 2, 1, 3)
      let weights = (MLXRandom.normal([8, 16, 3, 3]) * 0.03).asType(.float16)
      let layer = FrameGenerationFusedConv.Layer(weight: weights, bias: MLXArray.zeros([8], dtype: .float16))
      let a = FrameGenerationFusedConv.apply(FrameGenerator.upsample2(input), layer, activation: false)
      let b = FrameGenerationFusedConv.apply(input, layer, activation: false, upsample: true)
      eval(a, b)
      XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self))
    }
  }

  func testRealGeneratorParityAndTimingWhenConfigured() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["MLXDLSS_FG_UPSAMPLE_TIMING"] == "1", let path = env["MLXDLSS_FG_WEIGHTS"] else { throw XCTSkip("Set MLXDLSS_FG_UPSAMPLE_TIMING=1 and FG weights") }
    let weights = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    let original = FrameGenerationFusedConv.fusedUpsampleEnabled
    defer { FrameGenerationFusedConv.fusedUpsampleEnabled = original }
    for (height, width) in [(540, 960), (1080, 1920)] {
      for batch in [1, 4] {
        let a = MLXRandom.uniform(0..<1, [batch, height, width, 3])
        let b = MLXRandom.uniform(0..<1, [batch, height, width, 3])
        eval(a, b)
        var reference: MLXArray?
        for fused in [false, true, false] {
          FrameGenerationFusedConv.fusedUpsampleEnabled = fused
          let generator = try FrameGenerator(weights: weights)
          let phases = [Float](repeating: 0.5, count: batch)
          let output = try generator.interpolate(a, b, phases: phases)
          eval(output)
          if let reference { XCTAssertEqual(output.asArray(Float.self), reference.asArray(Float.self)) }
          else { reference = output }
          var samples: [Double] = []
          for i in 0..<24 {
            let start = ContinuousClock.now
            eval(try generator.interpolate(a, b, phases: phases))
            let t = start.duration(to: .now).components
            if i >= 4 { samples.append(Double(t.seconds) * 1000 + Double(t.attoseconds) / 1e15) }
          }
          print("FG-full \(batch)x\(height)x\(width) fused-upsample=\(fused): \(samples.sorted()[samples.count / 2]) ms")
        }
      }
    }
  }

  func testUpsampleConvolutionWhenConfigured() throws {
    let env = ProcessInfo.processInfo.environment
    guard env["MLXDLSS_FG_UPSAMPLE_TIMING"] == "1", let path = env["MLXDLSS_FG_WEIGHTS"] else { throw XCTSkip("Set MLXDLSS_FG_UPSAMPLE_TIMING=1 and FG weights") }
    let arrays = try loadArrays(url: URL(fileURLWithPath: path), stream: .cpu)
    for (block, height, width) in [(0, 68, 120), (1, 136, 240), (1, 272, 480)] {
      let w = arrays["block\(block).bot1.weight"]!, bias = arrays["block\(block).bot1.bias"]!
      let cin = w.dim(1)
      let spans = [0..<4, 4..<5, 5..<8]
      let rows = spans.enumerated().map { index, span in
        padded(w[span], widths: [[0, 0], [index * cin, (2 - index) * cin], [0, 0], [0, 0]])
      }
      let layer = FrameGenerationFusedConv.Layer(weight: concatenated(rows, axis: 0), bias: bias)
      for batch in [1, 4] {
        let input = (MLXRandom.normal([batch, height, width, cin * 3]) * 0.3).asType(.float16)
        func run(_ fused: Bool) -> MLXArray {
          FrameGenerationFusedConv.apply(fused ? input : FrameGenerator.upsample2(input), layer, activation: false, upsample: fused)
        }
        let a = run(false), b = run(true)
        eval(a, b)
        let d = abs(a.asType(.float32) - b.asType(.float32))
        print("upsample-conv parity block\(block) \(batch)x\(height)x\(width): max \(d.max().item(Float.self))")
        XCTAssertEqual(d.max().item(Float.self), 0)
        for fused in [false, true, false] {
          var samples: [Double] = []
          for i in 0..<24 {
            let start = ContinuousClock.now
            eval(run(fused))
            let t = start.duration(to: .now).components
            if i >= 4 { samples.append(Double(t.seconds) * 1000 + Double(t.attoseconds) / 1e15) }
          }
          print("upsample-conv block\(block) \(batch)x\(height)x\(width) fused=\(fused): \(samples.sorted()[samples.count / 2]) ms")
        }
      }
    }
  }
}
