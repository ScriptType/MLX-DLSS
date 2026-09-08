import CryptoKit
import Foundation
import MLX

/// Experimental DLSS SR K, LDR, 2x. One instance owns one sequential video stream.
/// Weights and the locally converted reconstruction shader are supplied separately.
public final class DLSSSuperResolver {
  public static let modelIdentifier = "dlss-sr-310.7.0-k-ldr-2x"
  static let postprocessDigest = "1bd949170193f077f75f9c08f8a4a9a6abf21088efbd7795ad8cd463be3517ae"
  static let channels = [32, 64, 64, 96, 128, 160, 128, 96, 64, 64, 32]
  static let heads = [2, 2, 2, 4, 4, 8, 4, 4, 2, 2, 2]

  public static let weightShapes: [String: [Int]] = {
    var result = [String: [Int]]()
    for index in 0..<11 {
      let stage = index + 1
      let c = channels[index]
      let h = heads[index]
      func add(_ name: String, _ shape: [Int]) { result["\(stage).\(name)"] = shape }
      if stage == 1 {
        add("embedding", [16, 32])
        add("embeddingBias", [32])
      }
      if stage >= 7 {
        add("embedding", [channels[index - 1], 4 * c])
        add("embeddingBias", [4 * c])
      }
      add("norm", [c])
      for head in 0..<h {
        for key in ["q", "k", "v"] { add("\(key)\(head)", [c, 32]) }
        add("position\(head)", [64, 64])
        add("projection\(head)", [32, c])
      }
      add("attentionBias", [c])
      add("ffScale", [c])
      add("ffOutputBias", [c])
      add("ffUp", [c, 4 * c])
      add("ffBias", [4 * c])
      add("ffDown", [4 * c, c])
      if stage <= 5 {
        let padded = (channels[index + 1] + 16 * h - 1) / (16 * h) * (16 * h)
        add("merge", [4 * c, padded])
        add("mergeBias", [padded])
      }
      if stage == 11 {
        add("output", [32, 48])
        add("outputBias", [48])
      }
    }
    result["reconstructionFilter"] = [32768]
    return result
  }()

  private struct Manifest: Decodable {
    let format: String
    let model: String
    let postprocessSHA256: String
  }
  private let weights: [String: MLXArray]
  private let postprocess: MLXFast.MLXFastKernel
  private var history: (color: MLXArray, luma: MLXArray, hidden: MLXArray)?
  private var dimensions: [Int] = []
  private var frameIndex = 0

  public init(packageURL: URL) throws {
    let manifest = try JSONDecoder().decode(
      Manifest.self,
      from: Data(contentsOf: packageURL.appendingPathComponent("manifest.json")))
    guard manifest.format == "mlxdlss-sr-v1", manifest.model == Self.modelIdentifier,
      manifest.postprocessSHA256 == Self.postprocessDigest
    else { throw MLXMediaError("Unsupported DLSS SR package") }
    let source = try Data(contentsOf: packageURL.appendingPathComponent("postprocess.metal"))
    let digest = SHA256.hash(data: source).map { String(format: "%02x", $0) }.joined()
    guard digest == Self.postprocessDigest, let text = String(data: source, encoding: .utf8) else {
      throw MLXMediaError("DLSS SR reconstruction shader does not match the supported conversion")
    }
    let sections = text.components(separatedBy: "// END HEADER")
    guard sections.count == 2 else { throw MLXMediaError("Invalid DLSS SR reconstruction shader") }
    let arrays = try loadArrays(
      url: packageURL.appendingPathComponent("weights.safetensors"), stream: .cpu)
    guard Set(arrays.keys) == Set(Self.weightShapes.keys) else {
      throw MLXMediaError("DLSS SR weight names do not match the supported model")
    }
    var prepared = [String: MLXArray]()
    for (name, shape) in Self.weightShapes {
      let value = arrays[name]!
      guard value.shape == shape, value.dtype == .float16 || value.dtype == .float32 else {
        throw MLXMediaError("Invalid DLSS SR tensor \(name); expected floating-point \(shape)")
      }
      let half = value.asType(.float16)
      guard isFinite(half).all().item(Bool.self) else {
        throw MLXMediaError("Nonfinite DLSS SR tensor: \(name)")
      }
      prepared[name] = half
    }
    weights = prepared
    postprocess = MLXFast.metalKernel(
      name: "mlxdlss_sr_reconstruct_v1",
      inputNames: [
        "net", "params", "filter", "color", "previousColor", "previousLuma", "motion", "depth",
        "dims",
      ],
      outputNames: ["rgb", "history", "luma", "hidden"], source: sections[1], header: sections[0])
  }

  public func reset() {
    history = nil
    dimensions = []
    frameIndex = 0
  }

  /// RGB [1,H,W,3] in [0,1]; motion [1,H,W,2] in current-to-previous input pixels.
  /// Returns evaluated [1,2H,2W,3] RGB. Reset at scene cuts, seeks and discontinuities.
  public func upscale(_ rgb: MLXArray, motion: MLXArray? = nil, reset: Bool = false) throws
    -> MLXArray
  {
    guard rgb.ndim == 4, rgb.dim(0) == 1, rgb.dim(3) == 3,
      rgb.dim(1) > 0, rgb.dim(2) > 0, rgb.dim(1) <= 4096, rgb.dim(2) <= 4096,
      rgb.dtype == .float16 || rgb.dtype == .float32
    else {
      throw MLXMediaError(
        "DLSS SR expects [1,H,W,3] floating-point RGB, up to 4096 pixels per side")
    }
    let h = rgb.dim(1)
    let w = rgb.dim(2)
    let ow = w * 2
    let oh = h * 2
    if let motion {
      guard motion.shape == [1, h, w, 2], motion.dtype == .float16 || motion.dtype == .float32,
        isFinite(motion).all().item(Bool.self)
      else {
        throw MLXMediaError(
          "DLSS SR motion must be finite [1,H,W,2] floating-point input-pixel offsets")
      }
    }
    guard isFinite(rgb).all().item(Bool.self) else {
      throw MLXMediaError("DLSS SR RGB must be finite")
    }
    let geometry = Self.geometry(width: w, height: h)
    let nw = geometry[4]
    let nh = geometry[5]
    let hw = geometry[6]
    let hh = geometry[7]
    if reset || dimensions != geometry { self.reset() }
    dimensions = geometry
    let first = history == nil
    let previous =
      history ?? (zeros([hh, hw, 4]), zeros([hh * 2, hw * 2, 1]), zeros([nh, nw, 4]))
    let color = concatenated(
      [clip(rgb.asType(.float32), min: 0, max: 1), ones([1, h, w, 1])], axis: 3
    ).squeezed(axis: 0)
    let flow = concatenated(
      [motion?.asType(.float32) ?? zeros([1, h, w, 2]), zeros([1, h, w, 2])], axis: 3
    ).squeezed(axis: 0)
    let depth = MLXArray.full([h, w, 4], values: MLXArray(Float(0.5)))
    let shape = MLXArray([
      UInt32(w), UInt32(h), UInt32(ow), UInt32(oh), UInt32(nw), UInt32(nh), first ? 1 : 0,
      UInt32(hw), UInt32(hh),
    ])
    let controls = MLXArray([Float(0), 0, 2, 2, 0, 0, 1])
    let prepared = Self.preprocess(
      [color, previous.0, previous.2, flow, depth, shape, controls],
      grid: (nw * nh, 1, 1), threadGroup: (256, 1, 1), outputShapes: [[nh, nw, 16]],
      outputDTypes: [.float16])[0]
    let features = network(prepared, Self.specifications(width: nw, height: nh, frame: frameIndex))
    let params = MLXArray(Self.parameters(geometry, reset: first))
    let output = postprocess(
      [
        features, params, weights["reconstructionFilter"]!, color, previous.0, previous.1, flow,
        depth, MLXArray(geometry.map(UInt32.init)),
      ],
      grid: (hw, hh, 1), threadGroup: (16, 16, 1),
      outputShapes: [[oh, ow, 4], [hh, hw, 4], [hh * 2, hw * 2, 1], [nh, nw, 4]],
      outputDTypes: [.float32, .float16, .float16, .float16], initValue: 0)
    eval(output)
    history = (output[1].asType(.float32), output[2].asType(.float32), output[3].asType(.float32))
    eval(history!.color, history!.luma, history!.hidden)
    frameIndex = (frameIndex + 1) % 8
    let result = contiguous(output[0][0..., 0..., 0..<3].expandedDimensions(axis: 0))
    eval(result)
    return result
  }

  static func geometry(width: Int, height: Int) -> [Int] {
    [
      width, height, width * 2, height * 2, max(256, (width * 2 + 127) / 128 * 32),
      max(256, (height * 2 + 127) / 128 * 32),
      (width * 2 + 15) / 16 * 16, (height * 2 + 15) / 16 * 16,
    ]
  }

  struct Spec {
    let stage: Int, width: Int, height: Int, offsetX: Int, offsetY: Int
    let gridX: Int, gridY: Int, channels: Int, heads: Int
  }

  static func specifications(width: Int, height: Int, frame: Int) -> [Spec] {
    let phases = [(0, 2), (5, 6), (4, 0), (1, 4), (7, 5), (2, 1), (3, 7), (6, 3)]
    var offsets = [(Int, Int)](repeating: (0, 0), count: 6)
    offsets[5] = phases[frame % 8]
    for index in stride(from: 4, through: 0, by: -1) {
      offsets[index] = ((offsets[index + 1].0 * 2 + 4) % 8, (offsets[index + 1].1 * 2 + 4) % 8)
    }
    return (0..<11).map { index in
      let level = min(index, 10 - index)
      let w = width >> level
      let h = height >> level
      let (x, y) = offsets[level]
      return Spec(
        stage: index + 1, width: w, height: h, offsetX: x, offsetY: y,
        gridX: (w + x + 7) / 8, gridY: (h + y + 7) / 8, channels: channels[index],
        heads: heads[index])
    }
  }

  static func parameters(_ geometry: [Int], reset: Bool) -> [UInt8] {
    let w = geometry[0]
    let h = geometry[1]
    let ow = geometry[2]
    let oh = geometry[3]
    var bytes = [UInt8](repeating: 0, count: 328)
    func put<T: FixedWidthInteger>(_ offset: Int, _ value: T) {
      var value = value.littleEndian
      withUnsafeBytes(of: &value) { bytes.replaceSubrange(offset..<(offset + $0.count), with: $0) }
    }
    func f(_ offset: Int, _ value: Float) { put(offset, value.bitPattern) }
    let sx = Float(w) * (1 / Float(ow))
    let sy = Float(h) * (1 / Float(oh))
    f(0, 1 / sx)
    f(4, 1 / sy)
    for offset in [16, 20, 24, 28] { f(offset, -0.5) }
    bytes[32] = reset ? 1 : 0
    for (offset, x, y) in [
      (40, sx, sy), (48, sx * 0.5, sy * 0.5), (56, sx, sy), (64, Float(0.5), Float(0.5)),
    ] {
      f(offset, x)
      f(offset + 4, y)
    }
    put(76, UInt32.max)
    put(80, UInt32(geometry[4]))
    put(84, UInt32(geometry[5]))
    for offset in [104, 120, 216] {
      put(offset, UInt32(w - 1))
      put(offset + 4, UInt32(h - 1))
    }
    put(136, UInt32(ow))
    put(140, UInt32(oh))
    f(144, 1 / Float(geometry[6]))
    f(148, 1 / Float(geometry[7]))
    f(152, 1 / Float(ow))
    f(156, 1 / Float(oh))
    f(160, 1)
    f(164, 1)
    f(168, 1)
    f(172, 1.37)
    f(176, 0.02)
    f(180, Float(Float16(0.02)))
    for (offset, value) in [(184, Float(0.04)), (186, 1 / 0.96), (188, 8), (190, -32), (192, -32)] {
      put(offset, Float16(value).bitPattern)
    }
    for (offset, handle) in [
      (200, 1), (224, 2), (232, 3), (240, 4), (248, 5), (288, 7), (296, 8), (304, 9), (312, 6),
    ] { put(offset, UInt64(handle)) }
    put(272, UInt64(1) << 32)
    put(280, UInt64(2) << 32)
    return bytes
  }
  private func half(_ x: MLXArray) -> MLXArray { x.asType(.float16) }
  private func mm(_ x: MLXArray, _ w: MLXArray, _ bias: MLXArray? = nil) -> MLXArray {
    var y = bias ?? zeros([x.dim(0), x.dim(1), w.dim(-1)], dtype: .float16)
    for k in stride(from: 0, to: x.dim(-1), by: 8) {
      let a = x[0..., 0..., k..<(k + 8)].asType(.float32)
      let b = (w.ndim == 2 ? w[k..<(k + 8), 0...] : w[0..., k..<(k + 8), 0...]).asType(.float32)
      y = half(matmul(a, b) + y.asType(.float32))
    }
    return y
  }
  private func reduceHalf(_ x: MLXArray) -> MLXArray {
    var s = half(x[0..., 0..., 0..<16] + x[0..., 0..., 16..<32])
    for i in stride(from: 32, to: x.dim(-1), by: 32) {
      s = half(s + half(x[0..., 0..., i..<(i + 16)] + x[0..., 0..., (i + 16)..<(i + 32)]))
    }
    s = half(s[0..., 0..., 0..<8] + s[0..., 0..., 8..<16])
    let a = half(s[0..., 0..., 0..<2] + s[0..., 0..., 2..<4])
    let b = half(s[0..., 0..., 4..<6] + s[0..., 0..., 6..<8])
    s = half(a + b)
    return half(s[0..., 0..., 0..<1] + s[0..., 0..., 1..<2])
  }
  private func normalized(_ x: MLXArray, _ gamma: MLXArray) -> MLXArray {
    let sum = reduceHalf(half(x * x)).asType(.float32)
    let inverse = half(which(sum .> 0, rsqrt(sum), MLXArray(Float(0))))
    return half(x * half(inverse * gamma))
  }
  private func probabilities(_ x: MLXArray) -> MLXArray {
    let t = clip(x, min: -21.640625, max: 21.640625)
    let factor = half(half(t * t).asType(.float32) * -0.0007119178771972656 + 1)
    let power = half(t.asType(.float32) * factor.asType(.float32) - 5.05078125)
    let e = half(pow(2, power.asType(.float32)))
    return half(e * half(1 / reduceHalf(e).asType(.float32)))
  }
  private func activation(_ x: MLXArray) -> MLXArray {
    let t = clip(x, min: -2, max: 2).asType(.float32)
    let slope = half(
      MLXArray(Float(Float16(Float(bitPattern: 0x3ed3_06eb)))) - MLXArray(
        Float(Float16(Float(bitPattern: 0x3da6_0dd6)))) * abs(t))
    return half(x * half(0.5 + t * slope.asType(.float32)))
  }
  private func windows(_ image: MLXArray, _ s: Spec, scale: Int = 1) -> MLXArray {
    let width = s.width / scale
    let height = s.height / scale
    let side = 8 / scale
    var indices = [Int32]()
    indices.reserveCapacity(s.gridX * s.gridY * side * side)
    for by in 0..<s.gridY {
      for bx in 0..<s.gridX {
        for dy in 0..<side {
          for dx in 0..<side {
            let yy = abs((by * 8 - s.offsetY) / scale + dy)
            let xx = abs((bx * 8 - s.offsetX) / scale + dx)
            let y = min(yy, 2 * (height - 1) - yy)
            let x = min(xx, 2 * (width - 1) - xx)
            indices.append(Int32(y * width + x))
          }
        }
      }
    }
    return image.reshaped(-1, image.dim(-1)).take(MLXArray(indices), axis: 0).reshaped(
      s.gridX * s.gridY, side * side, image.dim(-1))
  }
  private func image(_ tiles: MLXArray, _ s: Spec, scale: Int = 1) -> MLXArray {
    let side = 8 / scale
    let c = tiles.dim(-1)
    let padded = tiles.reshaped(s.gridY, s.gridX, side, side, c).transposed(0, 2, 1, 3, 4).reshaped(
      s.gridY * side, s.gridX * side, c)
    return padded[
      (s.offsetY / scale)..<((s.offsetY + s.height) / scale),
      (s.offsetX / scale)..<((s.offsetX + s.width) / scale), 0...]
  }
  private func network(_ initial: MLXArray, _ specs: [Spec]) -> MLXArray {
    var input = initial
    var skips = [Int: MLXArray]()
    for s in specs {
      let c = s.channels
      let n = s.gridX * s.gridY
      func w(_ key: String) -> MLXArray { weights["\(s.stage).\(key)"]! }
      var x: MLXArray
      if s.stage >= 7 {
        let low = windows(input, s, scale: 2)
        let up = mm(low, w("embedding"), w("embeddingBias")).reshaped(n, 4, 4, 2, 2, c).transposed(
          0, 1, 3, 2, 4, 5
        ).reshaped(n, 64, c)
        let skip = skips[12 - s.stage]!
        x = half(up + windows(skip, s))
      } else {
        x = windows(input, s)
        if s.stage == 1 { x = maximum(mm(x, w("embedding"), w("embeddingBias")), 0) }
      }
      let norm = normalized(x, w("norm"))
      let earlyResidual = (s.stage == 1 || s.stage == 11)
      var y = earlyResidual ? half(x + w("attentionBias")) : w("attentionBias")
      for head in 0..<s.heads {
        let v = mm(norm, w("v\(head)"))
        var p = w("position\(head)")
        if ![1, 2, 11].contains(s.stage) {
          let q = mm(norm, w("q\(head)"))
          let k = mm(norm, w("k\(head)"))
          p = probabilities(mm(q, k.transposed(0, 2, 1), p))
        }
        p = broadcast(p, to: [n, 64, 64])
        y = mm(mm(p, v), w("projection\(head)"), y)
      }
      if !earlyResidual { y = half(y + x) }
      let ff = activation(mm(half(y * w("ffScale")), w("ffUp"), w("ffBias")))
      y = mm(ff, w("ffDown"), half(y + w("ffOutputBias")))
      if s.stage <= 5 {
        skips[s.stage] = image(y, s)
        let folded = y.reshaped(n, 4, 2, 4, 2, c).transposed(0, 1, 3, 2, 4, 5).reshaped(
          n, 16, 4 * c)
        let merged = mm(folded, w("merge"), w("mergeBias"))
        let co = specs[s.stage].channels
        let compact = merged.reshaped(n, 16, s.heads, -1)[0..., 0..., 0..., 0..<(co / s.heads)]
          .reshaped(n, 16, co)
        input = image(compact, s, scale: 2)
      } else if s.stage == 11 {
        input = image(mm(y, w("output"), w("outputBias"))[0..., 0..., 0..<40], s)
      } else {
        input = image(y, s)
      }
      eval(input)
    }
    return input
  }

  private static let preprocess = MLXFast.metalKernel(
    name: "mlxdlss_sr_prepare_v1",
    inputNames: ["color", "history", "hidden", "motion", "depth", "shape", "options"],
    outputNames: ["result"],
    source: #"""
      uint id=thread_position_in_grid.x;
      uint2 lowSize(shape[0],shape[1]),outputSize(shape[2],shape[3]),netSize(shape[4],shape[5]);
      if(id>=netSize.x*netSize.y) return;
      float2 node(id%netSize.x,id/netSize.x),scale=float2(lowSize)/float2(outputSize);
      uint2 historySize(shape[7],shape[8]);
      float2 jitter(options[0],options[1]),motionScale(options[2],options[3]),motionBias(options[4],options[5]);
      float exposure=options[6];bool reset=shape[6]!=0;
      float currents[4],histories[4],rejections[4];float4 features[4];
      uint chosen[4]={1,3,0,2};
      for(uint q=0;q<4;q++) {
          float2 start=node*4.f+float2(q%2,q/2)*2.f;
          float3 old[4];float2 flow[4];bool valid=!reset;
          for(uint j=0;j<4;j++) {
              float2 pixel=start+float2(j%2,j/2);
              flow[j]=(sr_flow(motion,depth,sr_nearest((pixel+.5f)*scale-.5f+jitter),lowSize)+motionBias)*motionScale;
              float2 p=pixel+.5f+flow[j];valid=valid&&all(p>=0.f)&&all(p<=float2(outputSize));
              old[j]=sr_linear(history,p,historySize).xyz;
          }
          float3 mean(0),minimum(INFINITY),maximum(-INFINITY);
          int2 center=sr_nearest((start+1.f)*scale-.5f+jitter);
          for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
              float3 sample=sr_ycocg(sr_point(color,center+int2(dx,dy),lowSize));
              mean+=sample;float3 expanded=sample*(1.f+3.f*sample.x);
              minimum=min(minimum,expanded);maximum=max(maximum,expanded);
          }
          mean/=9.f;float3 spread=maximum-minimum;float rejection=0.f;
          for(uint j=0;j<4;j++) {
              float3 difference=abs(mean*(1.f+3.f*mean.x)-old[j]*(1.f+3.f*old[j].x));
              float3 outside=max(difference-.5f*spread,0.f)/(1.f+8.f*spread);
              rejection+=min(2.f*outside.x+.5f*outside.y+.5f*outside.z,1.f);
          }
          uint sample=chosen[q];float2 pixel=start+float2(sample%2,sample/2);
          float current=sr_ycocg(sr_point(color,sr_nearest((pixel+.5f)*scale-.5f+jitter),lowSize)).x;
          currents[q]=sr_compress(current,exposure);
          histories[q]=sr_compress(valid?old[sample].x:current,exposure);
          rejections[q]=valid?rejection*.25f:0.f;
          features[q]=valid?sr_linear(hidden,((node+.5f)*4.f+flow[sample])*.25f,netSize):float4(0);
      }
      float4 averaged=((features[0]+features[3])+(features[1]+features[2]))*.25f;
      for(uint q=0;q<4;q++) {
          result[id*16+q*4]=half(currents[q]);result[id*16+q*4+1]=half(histories[q]);
          result[id*16+q*4+2]=half(averaged[q]);result[id*16+q*4+3]=half(rejections[q]);
      }
      """#,
    header: #"""
      float4 sr_point(const device float* image, int2 p, uint2 size) {
          p=clamp(p,int2(0),int2(size)-1);
          return ((const device float4*)image)[p.y*size.x+p.x];
      }
      float4 sr_linear(const device float* image, float2 p, uint2 size) {
          p-=0.5f; int2 i=int2(floor(p)); float2 t=rint((p-float2(i))*256.f)/256.f;
          return mix(mix(sr_point(image,i,size),sr_point(image,i+int2(1,0),size),t.x),
                     mix(sr_point(image,i+int2(0,1),size),sr_point(image,i+1,size),t.x),t.y);
      }
      int2 sr_nearest(float2 p) { return int2(trunc(p+copysign(float2(0.5f),p))); }
      float3 sr_ycocg(float4 color) {
          float3 c=max(color.xyz,float3(0));
          return float3(c.x*.25f+c.y*.5f+c.z*.25f,(c.x-c.z)*.5f,c.y*.5f-c.x*.25f-c.z*.25f);
      }
      float sr_compress(float x,float exposure) {
          float z=x/max(1.f-x,.2f)*exposure;
          float y=clamp(z*max(1.f/(z+1.f),.0001f),0.f,1.f);
          return y>.0031308f ? 1.055f*pow(y,1.f/2.4f)-.055f : 12.92f*y;
      }
      float2 sr_flow(const device float* motion,const device float* depth,int2 p,uint2 size) {
          float center=sr_point(depth,p,size).x;
          int2 corners[4]={int2(-1,-1),int2(1,-1),int2(-1,1),int2(1,1)};
          float values[4];float nearest=center;int2 shift(0);
          for(uint i=0;i<4;i++) {
              values[i]=sr_point(depth,p+corners[i],size).x;
              if(values[i]<nearest) { nearest=values[i];shift=corners[i]; }
          }
          bool edge=center>center*.001f+(values[0]+values[3])*.5f || center>center*.001f+(values[1]+values[2])*.5f;
          return sr_point(motion,p+(edge?shift:int2(0)),size).xy;
      }
      """#)
}

/// Serializes temporal state and evaluates tensors before crossing actor boundaries.
public actor MLXNativeDLSSSuperResolver {
  private let model: DLSSSuperResolver
  public init(packageURL: URL) throws { model = try DLSSSuperResolver(packageURL: packageURL) }
  public func reset() { model.reset() }
  public func upscale(_ frame: MLXVideoFrame, motion: MLXVideoMotion? = nil, temporal: Bool = true)
    throws -> MLXVideoFrame
  {
    let pixels = motion.map { $0.vectors * MLXArray([Float(frame.width), Float(frame.height)]) }
    return MLXVideoFrame(
      try model.upscale(frame.array, motion: pixels, reset: !temporal || motion?.reset == true))
  }
}
