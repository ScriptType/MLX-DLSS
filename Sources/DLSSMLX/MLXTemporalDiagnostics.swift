#if MLXDLSS_TEMPORAL_DIAGNOSTICS
import Foundation
import DLSSCore
import MLX

/// Exact, contiguous native-dtype tensor bytes, exported only after frame completion.
public struct MLXTemporalDiagnosticTensor: Equatable, Sendable {
  public let shape: [Int]
  public let dtype: String
  public let bytes: Data

  init(_ array: MLXArray) {
    shape = array.shape
    dtype = String(describing: array.dtype)
    bytes = contiguous(array).asData(access: .copy).data
  }
}

public struct MLXTemporalDiagnosticSnapshot: Equatable, Sendable {
  public let frameIndex: UInt64
  public let noiseFrameIndex: UInt32
  public let logicalWidth: Int
  public let logicalHeight: Int
  public let networkWidth: Int
  public let networkHeight: Int
  public let tensors: [String: MLXTemporalDiagnosticTensor]
}

/// A value comparison of retained temporal state; descriptions are diagnostic Swift
/// representations, not a stable serialization protocol or a physical timing claim.
public struct MLXTemporalDiagnosticState: Equatable, Sendable {
  public let noiseFrameIndex: UInt32
  public let lifecycleDescription: String
  public let nativeCompositionDescription: String
  public let deviceFeaturesEnabled: Bool
  public let tensors: [String: MLXTemporalDiagnosticTensor]
}

public struct MLXTemporalDiagnosticReplay: Sendable {
  public let snapshot: MLXTemporalDiagnosticSnapshot
  public let fullResolutionModel: MLXVideoFrame
  public let stateBefore: MLXTemporalDiagnosticState
  public let stateAfter: MLXTemporalDiagnosticState
  public var stateUnchanged: Bool { stateBefore == stateAfter }
}

// These references never leave their owning renderer actor. Building a baseline
// capture retains existing immutable graphs without forcing evaluation/readback.
struct MLXTemporalDiagnosticArrays {
  let context: NeuralRenderFrameContext
  let geometry: NeuralRenderingNetworkGeometry
  let noiseIndex: UInt32
  let featureControls: NeuralRenderingFeatureControls
  let intensity: Float
  var tensors: [String: MLXArray]

  func exported() -> MLXTemporalDiagnosticSnapshot {
    MLXTemporalDiagnosticSnapshot(frameIndex: context.frameIndex, noiseFrameIndex: noiseIndex,
      logicalWidth: geometry.outputWidth, logicalHeight: geometry.outputHeight,
      networkWidth: geometry.networkWidth, networkHeight: geometry.networkHeight,
      tensors: tensors.mapValues { MLXTemporalDiagnosticTensor($0) })
  }
}
#endif
