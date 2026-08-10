# Architecture

SigOrbit combines a learned continuous pose estimator with a discrete
rotation-equivariant identity encoder.

## 1. Input contract

A cropped image is converted to grayscale, resized directly to an odd 257×257
square with Pillow bicubic interpolation, converted to float32 and mapped from
`[0,255]` to `[-1,1]`. There is no external deskew, thresholding, padding,
segmentation or reflection normalization.

The exact identifier is `sigorbit-gray-square-257-v1`. Changing any part of this
contract requires a new identifier and re-embedding every indexed reference.

## 2. SO(2) canonicalizer

A small localization CNN predicts two values and normalizes them to
`(cos θ, sin θ)`. They form a determinant-`+1` rotation matrix:

```text
[ cos θ   sin θ ]
[-sin θ   cos θ ]
```

`affine_grid` and bicubic `grid_sample` resample the input into the predicted
canonical orientation. The transform has no translation, scale, shear or
reflection terms. The last layer is initialized to `(1,0)`, so training begins
at the identity transformation.

Angle supervision is a training concern rather than an extra inference path.
At runtime the predicted pose always feeds the resampler, and the canonicalized
image always feeds the steerable backbone.

## 3. Steerable backbone (C_N, N = group_order)

The backbone uses `e2cnn` convolutions over the cyclic rotation group C_N,
where N is the configured `group_order` (4 or 8):

- stem: 7×7 R2Conv, inner BatchNorm, equivariant ReLU, antialiased /2 pooling;
- three stages: two 5×5 R2Conv blocks and antialiased /2 pooling;
- widths: 24, 48, 96 and 128 regular-representation fields;
- `GroupPooling` removes the C_N fiber;
- spatial average pooling and an MLP produce 256 values;
- L2 normalization produces the final descriptor.

C_N is not asked to approximate every input angle on its own. The continuous
canonicalizer removes most pose first; the steerable backbone then supplies a
strong, data-efficient rotation-aware feature extractor. C4 (90° symmetry) is
sufficient for signatures because the canonicalizer handles continuous rotation;
C8 (45° symmetry) provides finer equivariance at twice the parameter count and
training cost.

## 4. Training boundary

The companion `sigorbit-trainer` imports `ModelConfig`, `SteerableEncoder` and
`CanonicalizedEncoder` from the pinned runtime package. There is one architecture
implementation, not a trainer copy.

Optimization is split into three stages:

1. the backbone starts from random weights and learns signer identity with a
   temporary ArcFace classification head;
2. the best backbone and ArcFace state are restored, the backbone is frozen, and
   only the canonicalizer learns known synthetic rotation angles;
3. canonicalizer, backbone and ArcFace head are optimized together on clean and
   rotated views.

The exported artifact contains the canonicalizer and backbone only. ArcFace
weights, optimizer state and schedulers belong to recovery checkpoints and are
not needed for embeddings. The three stages are an optimization strategy; they
do not add branches or modes to the inference graph.

## 5. Why reflections are excluded

A mirrored signature can change stroke order and identity evidence. The model
uses SO(2)/C_N rotations rather than O(2)/D_N transformations, so it does not
make reflections equivalent by construction.

## 6. Artifact format

The release artifact contains only:

- `format_version`;
- strict architecture `config`;
- `model_state_dict`;
- non-executable metadata.

It removes e2cnn `.filter` and `.expanded_bias` caches because they are derived
from trained weights. This shrinks the artifact from 312,385,643 bytes to about
17 MiB with bit-identical embeddings. It is loaded using
`torch.load(..., weights_only=True)`; unexpected keys and missing learned keys
are rejected.

## 7. Parameter and compute profile

| | C4 | C8 |
|---|---:|---:|
| total parameters | 2,254,466 | 4,276,354 |
| backbone | 2,222,368 | 4,244,256 |
| canonicalizer | 32,098 | 32,098 |
| FP32 storage | ~9 MiB | ~17 MiB |

Output: 256 float32 values. The canonicalizer is identical for both group orders;
only the backbone's equivariant filter count scales with N.

Input resolution changes activation memory and latency, not parameter count.
For high-throughput deployment, replicate a complete model per GPU process and
batch requests; model parallelism is unnecessary.
