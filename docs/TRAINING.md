# Training recipe and reproducibility status

SigOrbit 0.1 focuses on a clean, verified inference package. The full training
CLI is being extracted from the research workspace and is not yet a supported
public API. This document records the selected run so the omission is explicit.

## Selected 257px run

```text
input size                 257
C8 widths                  24,48,96,128
embedding dimension        256
angle-only pretraining     10 epochs, lr 3e-3
joint training             40 epochs
rotation curriculum        ±10° → ±180° over 20 epochs
ArcFace margin             scheduled to 0.35
backbone lr                3e-4
canonicalizer lr           1e-3
orientation weight         0.5
consistency weight         0.5
workers                    6
```

The exact historical invocation was:

```bash
python finetune_canonicalized.py   --epochs 40 --in-size 257 --N 8   --backbone-ckpt checkpoints/equivariant_rotaug.pt   --out-dir checkpoints/257 --out-name canonicalized_257.pt   --dataset data/signature_dataset_disk   --canon-pretrain-epochs 10 --lr-canon-pretrain 3e-3   --curriculum-epochs 20 --rot-start 10 --rot-end 180   --lambda-orient 0.5 --lambda-consist 0.5   --lr-backbone 3e-4 --lr-canon 1e-3   --patience 999 --workers 6
```


## Dataset composition and licence status

The aggregate training split contained 1,320 CEDAR genuine images, 3,840
BHSig260 Hindi genuine images and 840 BHSig260 Bengali genuine images. Validation
(768 images) and test (792) were entirely BHSig260 Bengali. No forged signatures
were used.

The aggregate card says MIT, but CEDAR has no explicit licence grant and the
BHSig260 authors state only that the dataset is available “for research
purposes”. See `DATASET_LICENSE_AUDIT.md`. Do not publish a
retrained checkpoint from these inputs without upstream permission or counsel
approval.

## Loss design

1. **Identity:** ArcFace classification over training identities.
2. **Orientation:** direct circular supervision against `(cos α, sin α)` for the
   known synthetic rotation; clean images target `(1,0)`.
3. **Consistency:** `1 - cosine(e_clean, e_rotated)` rather than dimension-mean
   MSE, whose scale accidentally shrinks with a 256-D embedding.

The canonicalizer is pretrained separately because joint training from the
identity transform otherwise lets the backbone absorb augmentation while the
pose head remains close to zero.

## Reproducibility work required for 0.2

- publish the cleaned trainer and strict Pydantic/TOML experiment config;
- pin the exact dataset revision and generate a non-biometric synthetic smoke set;
- publish seed, optimizer/scheduler states and environment lock files;
- add checkpoint resume support;
- reproduce the selected run from random initialization;
- independently approve dataset and trained-weight provenance.

Do not claim full training reproducibility from the inference-only 0.1 release.
