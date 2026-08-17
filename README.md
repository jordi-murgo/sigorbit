# SigOrbit

[![CI](https://github.com/jordi-murgo/sigorbit/actions/workflows/ci.yml/badge.svg)](https://github.com/jordi-murgo/sigorbit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Rotation-robust handwritten-signature embeddings with continuous SO(2)
canonicalization and steerable C4/C8 backbones.**
Created and maintained by **Jordi Murgó**
([GitHub](https://github.com/jordi-murgo) ·
[jordi.murgo@gmail.com](mailto:jordi.murgo@gmail.com) ·
[jordi.murgo@gft.com](mailto:jordi.murgo@gft.com)).

SigOrbit turns a cropped signature image into a deterministic, L2-normalized
256-dimensional vector. It is designed for retrieval, clustering and downstream
verification systems that must tolerate arbitrary in-plane rotation without
making mirrored signatures equivalent.

> **Alpha release.** SigOrbit generates embeddings; it is not a complete fraud
> detector, legal signature verifier, document detector, identity database or
> anti-spoofing system. Do not use one model score as the sole basis for legal,
> financial or access-control decisions.

## Why the name?

A group action moves an input through its *orbit*. SigOrbit learns a canonical
SO(2) pose and embeds signatures consistently across that rotation orbit.
`sigorbit` was clear on PyPI and had no exact GitHub repository collision when
checked on 2026-08-07. This is not a trademark opinion.

## Architecture

````mermaid
flowchart TB
    subgraph Input["Preprocessing"]
        IMG["Cropped signature image"]
        GRAY["Grayscale → 257×257 bicubic → [-1, 1]"]
        IMG --> GRAY
    end

    subgraph Canon["OrientationCanonicalizer (SO(2))"]
        direction TB
        CC1["Conv2d 1→16 k5 s2 p2<br/>+ ReLU + BatchNorm2d"]
        CC2["Conv2d 16→32 k5 s2 p2<br/>+ ReLU + BatchNorm2d"]
        CC3["Conv2d 32→64 k3 s2 p1<br/>+ ReLU + BatchNorm2d"]
        CCP["AdaptiveAvgPool2d(1) → Flatten"]
        CLIN["Linear 64→2<br/>→ (cos θ, sin θ), L2-normalized"]
        CAFF["affine_grid + grid_sample<br/>(bicubic, rotation only)"]
        CC1 --> CC2 --> CC3 --> CCP --> CLIN --> CAFF
    end

    subgraph Backbone["SteerableEncoder (C4 or C8-steerable CNN, e2cnn)"]
        direction TB
        STEM["Stem<br/>R2Conv 1→24·N regular k7 p3<br/>+ InnerBatchNorm + ReLU<br/>+ BlurPool /2 (N = group_order)"]
        L1["Layer 1 (/4)<br/>R2Conv 24→48 k5 p2 + IBN + ReLU<br/>R2Conv 48→48 k5 p2 + IBN + ReLU<br/>+ BlurPool /2"]
        L2["Layer 2 (/8)<br/>R2Conv 48→96 k5 p2 + IBN + ReLU<br/>R2Conv 96→96 k5 p2 + IBN + ReLU<br/>+ BlurPool /2"]
        L3["Layer 3 (/16)<br/>R2Conv 96→128 k5 p2 + IBN + ReLU<br/>R2Conv 128→128 k5 p2 + IBN + ReLU<br/>+ BlurPool /2"]
        GP["GroupPooling<br/>max over C4/C8 fiber<br/>→ 128 invariant channels"]
        STEM --> L1 --> L2 --> L3 --> GP
    end

    subgraph Head["Embedding head"]
        direction TB
        POOL["AdaptiveAvgPool2d(1)<br/>→ Flatten → 128"]
        FC1["Linear 128→512<br/>+ BatchNorm1d + ReLU<br/>+ Dropout 0.3"]
        FC2["Linear 512→256<br/>+ BatchNorm1d"]
        NORM["L2-normalize"]
        POOL --> FC1 --> FC2 --> NORM
    end

    GRAY --> CC1
    CAFF --> STEM
    GP --> POOL
    NORM --> OUT["256-D L2-normalized embedding"]

    style Input fill:#1a2a3c,color:#fff
    style Canon fill:#1a3a5c,color:#fff
    style Backbone fill:#2d5a2d,color:#fff
    style Head fill:#4a3a1c,color:#fff
    style OUT fill:#5c1a1a,color:#fff
````

- C4: 2,254,466 trainable parameters (2.2 M); C8: 4,276,354 (4.3 M)
- 257×257 grayscale input
- 256-dimensional float32 output
- continuous SO(2) rotation canonicalization; no scale or reflection canonicalization
- C4 or C8 regular representations and invariant group pooling; the canonicalizer
  handles continuous rotation, so C4 equivariance (90° symmetry) is sufficient for
  signatures and trains 2.7× faster

See [the architecture notes](docs/ARCHITECTURE.md) and
[model card](docs/MODEL_CARD.md).

## Training and model lineage

The runtime package owns the exact model and preprocessing classes. The companion
[`sigorbit-trainer`](https://github.com/jordi-murgo/sigorbit-trainer) package
imports those classes directly rather than maintaining a second architecture.
Its current from-scratch protocol has three stages:

1. train the C4 or C8 backbone and a temporary ArcFace classifier;
2. restore the best backbone, freeze it, and pretrain only the SO(2)
   canonicalizer against known synthetic angles;
3. jointly fine-tune canonicalizer, backbone and ArcFace head with identity,
   circular-orientation and embedding-consistency losses.

The ArcFace head exists only during training and is not part of an exported
encoder. The published package still identifies the historically selected
checkpoint as `sigorbit-c8-257-v1`; that checkpoint used an older C8 initializer
whose complete resume history was not archived. The auditable trainer produces
two from-scratch model IDs: `sigorbit-c8-257-retrained-v1` (C8, batch 32) and
`sigorbit-c4-257-b64` (C4, batch 64). Neither claims a byte-for-byte
reproduction of the deployed model. See [training and reproducibility](docs/TRAINING.md)
for all lineages.

## Install

Python 3.10+ is supported. Install the correct PyTorch build for your CPU, CUDA
or ROCm platform first, then install SigOrbit:

```bash
python -m pip install sigorbit
# Optional FastAPI example:
python -m pip install "sigorbit[api]"
```

For an unreleased commit, installation directly from GitHub is also supported:

```bash
python -m pip install "sigorbit @ git+https://github.com/jordi-murgo/sigorbit.git@main"
```

For NVIDIA/ROCm, follow the official PyTorch selector rather than relying on the
CPU wheel chosen by a generic resolver.

## Python API

```python
from sigorbit import SignatureEncoder

encoder = SignatureEncoder(
    checkpoint="/secure/path/sigorbit-c8-257-v1.pt",
    device="auto",
)
vector = encoder.embed("signature.png")

print(vector.shape)  # (256,)
print(vector.dtype)  # float32
print(float(vector @ vector))  # ~1.0
print(encoder.model_id)  # sigorbit-c8-257-v1
print(encoder.preprocess_version)  # sigorbit-gray-square-257-v1
```

Batch inference preserves order:

```python
vectors = encoder.embed_batch(["a.png", "b.png"], batch_size=16)
assert vectors.shape == (2, 256)
```

Accepted inputs are `PIL.Image`, NumPy arrays, encoded bytes and file paths.
Inputs must already be cropped signatures. **Never compare embeddings generated
by different `model_id` or `preprocess_version` values.**

## FastAPI example

```bash
pip install -e '.[api]'
export SIGORBIT_CHECKPOINT=/secure/path/sigorbit-c8-257-v1.pt
export SIGORBIT_CHECKPOINT_SHA256=ec8d99f887f5a2658d93b14a14911b29a1411e9cf142efa85862a47b30cd233e
export SIGORBIT_API_KEY=replace-with-a-secret-from-your-secret-manager
SIGORBIT_DEVICE=auto sigorbit-api
```

Then, from an authorized client:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/embed \
  -H "Authorization: Bearer $SIGORBIT_API_KEY" \
  -F 'file=@signature.png'
```

The response includes the model/preprocess identity, predicted canonicalization
angle and 256 normalized floats. Interactive docs are at `/docs`.

Configuration:

| Variable | Default | Meaning |
|---|---|---|
| `SIGORBIT_CHECKPOINT` | required in code-only release | Local approved checkpoint path |
| `SIGORBIT_CHECKPOINT_SHA256` | required by HTTP API | Expected approved artifact digest |
| `SIGORBIT_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:0`, etc. |
| `SIGORBIT_API_KEY` | unset on loopback | Bearer token; required by CLI for non-loopback binding |
| `SIGORBIT_HOST` | `127.0.0.1` | Bind address |
| `SIGORBIT_PORT` | `8000` | Bind port |
| `SIGORBIT_MAX_UPLOAD_BYTES` | `10485760` | Encoded upload-file byte limit |
| `SIGORBIT_MAX_REQUEST_BYTES` | upload limit + 65536 | Pre-multipart request-body limit |
| `SIGORBIT_MAX_IMAGE_PIXELS` | `4194304` | Decompression-bomb pixel limit |
| `SIGORBIT_MAX_CONCURRENT_REQUESTS` | `1` | Full concurrent `/embed` request capacity |
| `SIGORBIT_QUEUE_TIMEOUT_SECONDS` | `1.0` | Wait before returning HTTP 503 |

Uploaded filenames are ignored: the API passes bytes to the decoder and never
constructs or writes a filesystem path from client input. Encoded inputs are
restricted to PNG, JPEG and WebP. The in-process Python API deliberately accepts
trusted `str`/`Path` inputs, so applications must not forward a remote filename
to `SignatureEncoder.embed()`.

See the [security review](docs/SECURITY_REVIEW.md) for the threat model,
adversarial tests and residual risks.

The server binds to loopback by default. Before binding to a non-loopback
address, configure bearer authentication and place it behind a TLS reverse proxy
with request/time/rate limits. The built-in limits reduce accidental and simple
resource exhaustion; they are not a replacement for an edge proxy or process
isolation.

The example intentionally has no signer database and no hard-coded MATCH
threshold. Thresholds are catalogue-, domain- and reference-count-specific.

## Measured behavior

On the held-out 33-signer/792-image **BHSig260 Bengali** test split, the
257px checkpoint achieved:

- clean leave-one-out top-1: **100.0% (792/792)**;
- clean median margin: **+0.3196**;
- model-canvas non-zero-angle mean top-1: **98.04%**;
- square-expand mean top-1: **98.16%**;
- raw-expand mean top-1: **89.87%**;
- real-signature all-triplet mean top-1: **87.64%**.

These numbers describe one dataset/protocol, not a universal error rate. See the
model card for exact protocols, limitations and the 129px comparison.

## Development

```bash
python -m pip install -e '.[api,dev]'
ruff check .
pytest
python -m build
```

Repository layout:

```text
src/sigorbit/model.py          architecture
src/sigorbit/checkpoint.py     safe artifact loading and identity
src/sigorbit/preprocessing.py  versioned image contract
src/sigorbit/encoder.py        public Python API
src/sigorbit/api.py            minimal FastAPI example
docs/                          architecture, model card, training and release notes
CITATION.bib                   project and technical bibliography
```

## Responsible use and release status

Handwritten signatures are biometric personal data. Do not log uploads or
embeddings unnecessarily, and establish retention, consent, access control and
deletion policies before deployment.

The source code is MIT licensed. The repository contains no training images.
Before publishing the trained checkpoint, read the completed
[dataset licence audit](docs/DATASET_LICENSE_AUDIT.md) and complete
[RELEASING.md](docs/RELEASING.md). All 7,560 aggregate images were matched
pixel-for-pixel to genuine CEDAR and BHSig260 samples. CEDAR has no explicit
licence grant, while the BHSig260 authors state only “for research purposes”. The Hub uploader's MIT tag therefore
does not establish a sublicence chain. The local checkpoint is present to
validate this extraction; **do not push or publish it** until upstream permission
or counsel approval is recorded.

## License and citation

Source code: [MIT](LICENSE). Third-party notices: [NOTICE](NOTICE).
Repository: <https://github.com/jordi-murgo/sigorbit>.

If SigOrbit is useful in research, cite the software as:

```bibtex
@software{murgo2026sigorbit,
  author  = {Jordi Murgó},
  title   = {{SigOrbit}: Rotation-Robust Handwritten-Signature Embeddings},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/jordi-murgo/sigorbit},
  license = {MIT}
}
```

The complete BibTeX bibliography—including the e2cnn, CEDAR and BHSig260
references—is in [`CITATION.bib`](CITATION.bib); machine-readable citation
metadata is in [`CITATION.cff`](CITATION.cff). Citing a dataset does not imply
that it has an open licence; consult the
[dataset licence audit](docs/DATASET_LICENSE_AUDIT.md).
