# Model card: `sigorbit-c8-257-v1`

## Summary

SigOrbit is an offline handwritten-signature embedding model. It accepts one
cropped signature and returns a 256-dimensional unit vector intended for cosine
retrieval or a separately calibrated decision system.

| Field | Value |
|---|---|
| Architecture | learned SO(2) canonicalizer + C8 steerable CNN |
| Input | 257×257 grayscale, bicubic square resize, `[-1,1]` |
| Output | 256-D float32, L2 normalized |
| Parameters | 4,276,354 |
| Reflections | deliberately not invariant |
| Semantic model ID | `sigorbit-c8-257-v1` |
| Preprocess ID | `sigorbit-gray-square-257-v1` |
| Slim artifact SHA-256 | `ff62889e2c5172762e894f4ee05162e0109d06c863b1fa83a7bcc6baf1dc0963` |
| Public weight release | withheld pending CEDAR/BHSig260 permission |
| Original training artifact SHA-256 | `39384c2d7211bb1815b7cd8f65ba15aca26b83b89b8206205eadd9b416954399` |

Slim and original artifacts produced bit-identical embeddings in the extraction
parity test (`max_abs_diff=0`, cosine=1 for the test crop).

## Training

The backbone was trained from random initialization on the dataset published as
`rakshitdabral/Signature-Verification-Dataset`; the canonicalized run initialized
from that C8 backbone. It does **not** inherit the historical HairyPotato
EfficientNet/GPL checkpoint used in unrelated experiments.

The selected 257px run used:

- 40 joint epochs after 10 angle-only canonicalizer pretraining epochs;
- synthetic rotation curriculum from ±10° to ±180° over 20 epochs;
- ArcFace identity loss;
- direct circular orientation supervision;
- cosine embedding-consistency loss;
- no reflected samples;
- selected checkpoint: zero-based epoch 33;
- validation top-1 100%, median margin +0.31266865.

A decoded-pixel audit proved that the aggregate contains every genuine image
from CEDAR (55×24) and BHSig260 (260×24), with no forgeries. The 6,000-image
training split contains all 55 CEDAR identities, all 160 BHSig260 Hindi
identities, and 35 of the 100 BHSig260 Bengali identities. CEDAR has no explicit
licence grant; the BHSig260 authors state only “for research purposes”. The aggregate Hub card's MIT tag
does not establish authority to relicense those images. No images are
redistributed here, and the checkpoint must not be published until upstream
permission or counsel approval is recorded. See `DATASET_LICENSE_AUDIT.md`.

## Evaluation protocol

The untouched test split contained 792 images from 33 BHSig260 Bengali
signers. It is identity-disjoint but not language-balanced: the validation and
test splits are Bengali-only because identities were split contiguously. Clean
evaluation used leave-one-out matching. Rotation sweeps used the first three
clean references for every signer and 693 remaining queries per angle at
0, 5, 10, 15, 20, 30, 45, 60, 90 and 180 degrees.

| Metric | Result |
|---|---:|
| Clean top-1 | 100.0% (792/792) |
| Clean median margin | +0.3196 |
| Fragile clean queries, margin <0.05 | 0.9% |
| Model-canvas mean top-1, non-zero angles | 98.04% |
| Square-expand mean top-1, non-zero angles | 98.16% |
| Raw-expand mean top-1, non-zero angles | 89.87% |
| Raw-expand mean margin, non-zero angles | +0.1636 |

Raw-expand top-1 at the hardest intermediate angles was 94.2% at 10°, 92.1% at
15°, 88.2% at 20°, 83.4% at 30°, 77.9% at 45° and 80.8% at 60°.

A separate nine-image real-signature study evaluated all 84 possible triplets of
three references against 315 catalogue identities. Mean top-1 over angles was
87.64%; the small, non-random sample is only an out-of-domain diagnostic.

## Intended uses

- feature extraction for cropped offline signatures;
- similarity search, deduplication and clustering;
- research on rotation-robust biometric representations;
- one component of a human-reviewed verification workflow.

## Out-of-scope uses

- detecting signatures in full documents;
- proving authenticity or forgery by itself;
- online/dynamic signature verification;
- anti-spoofing or presentation-attack detection;
- inferring identity without an enrolled, compatible reference set;
- autonomous legal, financial, employment or access-control decisions.

## Limitations and risks

1. **Domain dependence and split bias.** Accuracy may change with language,
   pen, scanner, compression, crop quality, background and demographic
   distribution. Reported validation/test metrics are Bengali-only.
2. **Framing is not rotation.** Expanding a rectangular canvas also changes
   scale/aspect context; performance is lower than pure model-canvas rotation.
3. **No reflection invariance.** This is intentional and must not be changed
   silently.
4. **Reference quality matters.** Diverse clean references are better than near
   duplicates; three references were the minimum evaluation configuration.
5. **Thresholds are not portable.** A production experiment used 0.83 similarity
   and 0.03 winner gap, but those values are not part of this embedding API and
   must not be copied without domain-specific open-set calibration.
6. **Biometric privacy.** Images and embeddings can be personal/biometric data.
   Embeddings are not anonymous and may leak information or support linkage.
7. **No forgery benchmark.** Identity retrieval accuracy is not a false-accept
   rate against skilled forgeries.

## Recommended validation before deployment

- lock model and preprocess IDs in every stored vector;
- measure known/unknown and skilled-forgery behavior on the target population;
- calibrate by reference count and report confidence intervals;
- test rotations, scale, crop, ink, scanner and compression perturbations;
- establish human review and rollback paths;
- document consent, retention, deletion and access controls;
- monitor MATCH/REVIEW/NO_MATCH distributions without storing raw signatures in
  general application logs.
