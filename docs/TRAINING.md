# Training and reproducibility

`sigorbit` is the inference package. Training lives in the companion
[`sigorbit-trainer`](https://github.com/jordi-murgo/SigOrbit-trainer) repository,
which pins `sigorbit==0.1.0` and imports its model classes directly. The trainer
source is published on GitHub as an alpha; it is not distributed on PyPI.

The distinction matters because there are two model lineages:

- `sigorbit-c8-257-v1` is the historically selected deployment checkpoint
  described by the model card;
- `sigorbit-c8-257-retrained-v1` is the auditable, from-scratch candidate produced
  by the current trainer. It is a separate model ID and has not replaced the
  historical checkpoint.

## Current three-stage protocol

The validated `c8-257-final` run uses one 257 px architecture throughout:
C8 widths `24,48,96,128`, 256-D embeddings and PK batches with 8 signers × 4
samples. All learned weights start randomly; no external initializer is loaded.

| Stage | Optimized parameters | Objective and augmentation | Schedule |
|---|---|---|---|
| **1. Backbone** | C8 encoder + train-only ArcFace head | signer classification; framing/color/affine jitter and one-sided expanded PIL rotations at 0°, 15°, 30°, 45°, 60° or 75° | 40 epochs, AdamW, 2-epoch LR warmup then cosine |
| **2. Pose** | SO(2) canonicalizer only | circular supervision for clean images and known randomly rotated copies; rotation envelope grows from ±45° to ±180° | 10 epochs, LR `3e-3` |
| **3. Joint** | canonicalizer + C8 encoder + ArcFace head | identity on clean and rotated views, circular pose loss and cosine embedding consistency; envelope grows from ±10° to ±180° over 20 epoch indices | 80 epochs, 5-epoch LR warmup then cosine |

Stage 2 starts from the best validation checkpoint of stage 1, not the last
backbone epoch. Stage 3 starts from that backbone and the final pose-pretraining
state. The final export restores the best joint checkpoint according to the
validation selection rule; it is not necessarily the epoch-80 state.

For a clean tensor \(x\), sampled angle \(\alpha\), rotated view \(R_\alpha x\),
signer label \(y\), embedding \(e\) and predicted unit pose \(p\):

```text
L_joint = L_arc + 0.5 L_orient + 0.5 L_consist

L_arc     = 0.5 [CE(ArcFace(e(x), y)) + CE(ArcFace(e(Rαx), y))]
L_orient  = circular pose loss for p(x)→0 and p(Rαx)→α
L_consist = 1 - cosine(e(x), e(Rαx))
```

The ArcFace scale is 16 and its angular margin is held at zero initially, then
ramped to 0.35. The joint stage uses separate learning rates: `3e-4` for the
backbone and `1e-3` for both canonicalizer and ArcFace head. The 80-epoch run
deliberately disables practical early stopping so the increasing rotation
difficulty and full cosine decay complete before checkpoint selection.

Training tensors and parameters remain FP32. The validated NVIDIA configuration
enables TF32 matrix/convolution kernels for speed and sets
`run.deterministic = false`: CUDA does not provide a deterministic backward
kernel for the bicubic `grid_sample` used by the canonicalizer.

## Why the stages are separate

Training everything jointly from random initialization gives the identity loss
two moving targets: an untrained pose transform and an untrained identity
encoder. The backbone can minimize identity loss by absorbing the current
augmentation while the identity-initialized canonicalizer stays near 0°.

The staged protocol removes that ambiguity:

1. the backbone first learns useful signer features;
2. the canonicalizer then learns an explicit angular task against a frozen
   identity representation;
3. joint training adapts both modules so embeddings remain stable across the
   complete rotation orbit without giving up signer separation.

The model used at inference is still a single serial graph:
canonicalizer → resampler → C8 backbone → normalized embedding. The stages add
no inference-time branches. The ArcFace classifier is discarded after training.

## Historical deployed checkpoint

The deployed `sigorbit-c8-257-v1` has the same inference architecture but a
different training history. Its C8 initializer came from a separate
discrete-rotation experiment that was itself resumed from an earlier checkpoint.
The canonicalized run then performed 10 angle-only epochs and 40 joint epochs;
zero-based joint epoch 33 was selected.

That artifact is valid deployment evidence, but its optimizer/scheduler history
was not archived well enough for a byte-identical reconstruction. The current
trainer therefore starts stage 1 from random weights and records the result as
an attempted reproduction rather than relabelling it as the deployed model.

## Running the auditable trainer

From a checkout of `SigOrbit-trainer`:

```bash
uv sync --extra dev
uv run sigorbit-train config validate configs/c8-257-final.toml
uv run sigorbit-train dataset validate \
  /secure/sigorbit-dataset/dataset.toml \
  --attestation /secure/sigorbit-dataset/rights.attestation.json
uv run sigorbit-train run configs/c8-257-final.toml
```

The trainer supports exact resume at completed epoch boundaries with strict
configuration, dataset, class-map, architecture, topology and RNG checks.
Recovery checkpoints include ArcFace, optimizer, scheduler and RNG state;
deployable exports contain only the runtime model and non-executable metadata.

See the trainer's
[`TRAINING_RECIPE.md`](https://github.com/jordi-murgo/SigOrbit-trainer/blob/main/docs/TRAINING_RECIPE.md),
[`REPRODUCIBILITY.md`](https://github.com/jordi-murgo/SigOrbit-trainer/blob/main/docs/REPRODUCIBILITY.md)
and
[`RESULTS_c8-257-final.md`](https://github.com/jordi-murgo/SigOrbit-trainer/blob/main/docs/RESULTS_c8-257-final.md)
for the exact configuration and observed result.

## Dataset composition and rights

The historical deployment aggregate contained 1,320 CEDAR genuine images and
6,240 BHSig260 genuine images; no forged samples were used. Its original split
had 6,000 training, 768 validation and 792 test images. The current trainer run
uses the decoded-pixel-deduplicated manifest: 5,939 / 610 / 580 images across
250 / 32 / 33 signer-disjoint train, validation and test splits.

CEDAR has no explicit licence grant and the BHSig260 authors state only that the
dataset is available “for research purposes”. The aggregate Hub card's MIT tag
does not establish authority to relicense those images or their trained weights.
No dataset or checkpoint is distributed by either source package. Do not publish
a retrained checkpoint without upstream permission or counsel approval; see
[`DATASET_LICENSE_AUDIT.md`](DATASET_LICENSE_AUDIT.md).
