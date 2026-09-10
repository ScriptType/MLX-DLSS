import MLX

/// Test-only candidate. The center remains Float32; only the predicate is separable.
enum SeparableMotionErosion {
  static func apply(_ input: MLXArray, params: MLXArray) -> MLXArray {
    precondition(input.ndim == 4 && input.dim(0) == 1 && input.dim(3) == 1 && input.dtype == .float32)
    let count = input.dim(1) * input.dim(2)
    let mask = horizontal([input, params], grid: (count, 1, 1),
      threadGroup: (256, 1, 1), outputShapes: [input.shape], outputDTypes: [.uint8])[0]
    return vertical([input, mask, params], grid: (count, 1, 1),
      threadGroup: (256, 1, 1), outputShapes: [input.shape], outputDTypes: [.float32])[0]
  }

  private static let horizontal = MLXFast.metalKernel(
    name: "mlxdlss_test_separable_erosion_horizontal", inputNames: ["input", "params"], outputNames: ["mask"],
    source: #"""
      uint i = thread_position_in_grid.x; int width = params[0], height = params[1];
      if (i >= uint(width * height)) return;
      int x = i % width, y = i / width;
      bool valid = true;
      for (int dx = -3; dx <= 3; ++dx)
        valid = valid && input[y * width + clamp(x + dx, 0, width - 1)] > 0.0f;
      mask[i] = valid ? uchar(1) : uchar(0);
      """#)

  private static let vertical = MLXFast.metalKernel(
    name: "mlxdlss_test_separable_erosion_vertical", inputNames: ["input", "mask", "params"], outputNames: ["confidence"],
    source: #"""
      uint i = thread_position_in_grid.x; int width = params[0], height = params[1];
      if (i >= uint(width * height)) return;
      int x = i % width, y = i / width;
      bool valid = true;
      for (int dy = -3; dy <= 3; ++dy)
        valid = valid && mask[clamp(y + dy, 0, height - 1) * width + x] != uchar(0);
      confidence[i] = valid ? input[i] : 0.0f;
      """#)
}
