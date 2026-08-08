# SigOrbit

[![CI](https://github.com/jordi-murgo/SigOrbit/actions/workflows/ci.yml/badge.svg)](https://github.com/jordi-murgo/SigOrbit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Rotation-robust handwritten-signature embeddings with continuous SO(2)
canonicalization and a C8-steerable backbone.**

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

```text
cropped image
    │ grayscale → 257×257 bicubic → [-1, 1]
    ▼
SO(2) canonicalizer
    │ predicts normalized (cos θ, sin θ)
    │ affine_grid + bicubic grid_sample; rotation only
    ▼
C8 steerable CNN (e2cnn)
    │ antialiased pooling → group pooling
    ▼
256-D L2-normalized embedding
```

- 4,276,354 trainable parameters
- 257×257 grayscale input
- 256-dimensional float32 output
- continuous rotation canonicalization; no scale or reflection canonicalization
- C8 regular representations and invariant group pooling
- slim-checkpoint tooling: 16.9 MiB locally; checkpoint intentionally excluded from public source builds pending permission

See [the architecture notes](docs/ARCHITECTURE.md) and
[model card](docs/MODEL_CARD.md).

## Install

Python 3.10+ is supported. Install the correct PyTorch build for your CPU, CUDA
or ROCm platform first, then install SigOrbit:

```bash
# from a clone
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[api]'
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
SIGORBIT_CHECKPOINT=/secure/path/sigorbit-c8-257-v1.pt \
SIGORBIT_DEVICE=auto sigorbit-api
```

Then:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/embed   -F 'file=@signature.png'
```

The response includes the model/preprocess identity, predicted canonicalization
angle and 256 normalized floats. Interactive docs are at `/docs`.

Configuration:

| Variable | Default | Meaning |
|---|---|---|
| `SIGORBIT_CHECKPOINT` | required in code-only release | Local approved checkpoint path |
| `SIGORBIT_DEVICE` | `auto` | `cpu`, `cuda`, `cuda:0`, etc. |
| `SIGORBIT_HOST` | `127.0.0.1` | Bind address |
| `SIGORBIT_PORT` | `8000` | Bind port |
| `SIGORBIT_MAX_UPLOAD_BYTES` | `10485760` | Upload limit |

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
Repository: <https://github.com/jordi-murgo/SigOrbit>.

If SigOrbit is useful in research, cite the software as:

```bibtex
@software{murgo2026sigorbit,
  author  = {Jordi Murgó},
  title   = {{SigOrbit}: Rotation-Robust Handwritten-Signature Embeddings},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/jordi-murgo/SigOrbit},
  license = {MIT}
}
```

The complete BibTeX bibliography—including the e2cnn, CEDAR and BHSig260
references—is in [`CITATION.bib`](CITATION.bib); machine-readable citation
metadata is in [`CITATION.cff`](CITATION.cff). Citing a dataset does not imply
that it has an open licence; consult the
[dataset licence audit](docs/DATASET_LICENSE_AUDIT.md).
