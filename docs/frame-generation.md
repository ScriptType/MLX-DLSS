# Frame generation: what was recovered and how it was verified

MLX-DLSS's frame generator is a port of the video path of NVIDIA's DLSS
Frame Generation library (`libnvidia-ngx-dlssg.so`, DLSS SDK 310.7.0). This
note records the recovered graph, the evidence behind each piece and the
numbers that tie the port to the vendor's output. Nothing proprietary is
reproduced here: the weights stay in the user's copy of the library and are
extracted locally with `mlxdlss-weights extract-fg`.

## Method

The library only exposes its frame generator through Vulkan (`VK_NVX_binary_import`
launches of CUDA kernels). A headless harness fed it plain video frames with
zero motion vectors and a flat depth, and a hook on the device dispatch table
recorded every kernel launch with its parameter block, every weight upload,
and memory snapshots of the intermediate buffers and images after each launch.
The kernels' PTX was recovered from the library's fat binaries and read
through a small expression printer that folds a kernel's straight-line code
into the expressions it stores. Every stage below was then implemented in
PyTorch and compared with the snapshot of the same buffer on the same frames.

## The graph (one evaluate, 75 launches)

For video the library's motion-vector machinery — motion-vector dilation,
depth-keyed forward splatting into the interpolated time, occlusion weights,
the mask U-Net that judges the splat — all runs but produces constants: the
splatted flows are zero, the occlusion weights are `sigmoid(0)`, and the
warped candidates are the frames themselves. What remains is:

| stage | operation | verification |
| --- | --- | --- |
| candidates | 2×2 box means of the two frames (zero-padded to a multiple of 16) | MAE 1e-4 against the library's candidate buffer |
| error channel | mean-RGB \|a − b\| at half resolution, fetched at full resolution with bilinear filtering and box-averaged back: a separable `[1/8, 3/4, 1/8]` blur | MAE 4e-5 |
| block input | `[a, err, b, err, mask = 0, phase t]`, 10 channels | layout read from `k_initial_merge` |
| block0 | three stem convs (3×3, LeakyReLU 0.01 clamped to ±6, 2×2 mean pool), eight residual convs in pairs, three activated heads on the ×2-upsampled trunk, one linear 8-channel head on the ×2-upsampled heads: flow A (2), flow B (2), mask logit, residual (3) | every layer 0.03–0.3 % against its snapshot |
| refinement input | the candidates warped by twice block0's flows, block0's outputs, the zero mask, the residual and the phase: 18 channels | 0.24 % |
| block1 | the same structure with 16/32 channels; `flow = 2·f0 + f1`, `mask = m0 + m1`, `res = r0 + r1` | export buffer within 1.1–1.7 % (flows), 3.7 % (mask), 0.5 % (residual), correlation 1.000 |
| output | the export upsampled bilinearly (plain half-scale sampling, index-clamped), `sigmoid(mask)`, both full-resolution frames warped by the scaled flows and blended by the mask; the residual is not used by the output kernel | the library's `OutputInterpFrame` at **59.9 dB PSNR**, maximum 3/255 |

The library composes at full resolution inside its `main_kernel` family; the
synthesis networks run at half resolution. The multi-frame mode is the same
graph evaluated at phases k/N.

Layouts worth knowing when reading the kernels: the custom convolutions keep
weights as `[co/8][tap][ci/8][co%8][ci%8]`, the U-Net's `k_conv_fp16_nhwc`
keeps `mma.m16n8k16` fragments with a permuted 16-column order and the bias in
the tail of the same blob, and activations are NHWC fp16 with 8×8 tiles. The
extractor undoes all of it into dense `[Cout, Cin, 3, 3]` tensors.

## Whole-clip results

The README's five clips, even frames in and odd frames withheld, PSNR against
the withheld frames, the vendor's library against this port (PyTorch, M2 Max):

| clip | library | port |
| --- | --- | --- |
| train | 27.38 | 27.39 |
| handheld | 30.90 | 30.91 |
| druid | 33.48 | 33.49 |
| muse | 32.11 | 32.13 |
| fishes | 38.89 | 38.92 |

## Speed

Current float16 Metal dispatch, M2 Max, real weights and nonconstant synthetic
RGB inputs. These paired warm measurements include synthesis and composition
but exclude process startup and I/O. Times below are for the **whole batch**;
each batch produces `pairs × (factor − 1)` generated frames.

| extent | pairs | factor | scalar | auto |
| --- | --- | --- | --- | --- |
| 960×540 | 1 | 2 | 5.30 ms | 5.30 ms |
| 960×540 | 4 | 2 | 17.78 ms | 16.47 ms |
| 960×540 | 4 | 4 | 49.17 ms | 45.76 ms |
| 1920×1080 | 1 | 2 | 17.51 ms | 16.58 ms |
| 1920×1080 | 4 | 2 | 65.16 ms | 60.68 ms |
| 1920×1080 | 8 | 2 | 131.38 ms | 124.28 ms |
| 1920×1080 | 4 | 3 | 127.93 ms | 122.18 ms |
| 1920×1080 | 4 | 4 | 192.01 ms | 186.51 ms |

Auto dispatch uses SIMD matrix convolutions for eligible unpooled layers
with at least 32 output channels and 16,384 batch-inclusive pixels. Pooled
stems stay scalar: forcing SIMD throughout the network was slower. Batch 4
remains the default; a larger batch does not consistently reduce time per
generated frame. Set `MLXDLSS_FG_SIMD=0` to force scalar or `=1` to use SIMD
where supported when comparing on another Apple GPU; leave it unset for auto.
On the tested real-weight frames, auto/scalar float16 differences stayed
below `3.9e-4` maximum and `7.1e-6` mean absolute error. The float32 path is
unchanged.

Earlier end-to-end and cross-backend measurements, per generated frame after
warm-up (kernel compilation excluded), float16 unless noted:

| | 960×540 | 1920×1080 |
| --- | --- | --- |
| Metal, `mlxdlss framegen`, one frame (N = 1) | 6.3 ms | 21 ms |
| Metal, `mlxdlss framegen --factor 4` (the three phases as one batch, N = 3) | 5.2 ms | 23 ms |
| Metal, `mlxdlss framegen-stream --batch 4` through the uint8 pipe (what `mlxdlss-video framegen --backend mlxdlss` uses) | 6.4 ms | 25 ms |
| Metal, `mlxdlss framegen-stream --batch 1`, float32 pipe (the previous protocol) | 27 ms | 43 ms |
| Metal (float32, MLX convolutions) | 9.2 ms | 32 ms |
| PyTorch / MPS (float16) | 5.3 ms | 17 ms |
| PyTorch / MPS (float32) | 6.3 ms | 22 ms |

Every stage of the Swift port takes a batch (`[N, H, W, 3]` frames, one phase
per sample): the phases of one pair (`--factor 3|4`) and the consecutive pairs
of a video (`--batch`, default 4) run as one pass. Batching does not lower the
GPU time per frame much (the layers are throughput-bound already at N = 1:
540p gains 17 % at N = 3, 1080p nothing), but it lets the frame server overlap
the host work with the GPU: with `--batch 1` the server waits for the host
after every frame.
Standalone FG uses uint8 RGB in both pipe directions (`--format u8`, the
default), which took the host side of the stream from 22 ms to under 2 ms per
540p frame; the frames are converted on the GPU. The web NR/FG chain uses
`--format f32` in either effect order to preserve fractional detail between
models, with one decode and one final encode.

Each convolution runs as one Metal kernel with the bias, the clamped
LeakyReLU, the residual add and the 2×2 mean pool in its epilogue (the three
heads as one convolution, the linear head as a block-diagonal one); each
block's input is assembled by a single kernel (box means, blurred error,
upsampled flows, warps) and the output composed by another. The PyTorch port
batches the same way (`FrameGenerator.generate` and `generate_pairs`).
Accuracy: the two ports agree within `1e-7` MAE at float32 and `1.3e-5` at
float16 on real weights (a full 960×540 frame); a batched result equals the
one-frame result exactly at float32 and within float16 rounding on MPS.

## Not ported

The motion-vector and depth inputs, the HUD-less/UI compositing and the
disocclusion inpainting pass exist in the library and are no-ops for plain
video, so the port leaves them out; the recovered kernel roles are recorded in
the research notes should an engine-integrated caller ever want them.
