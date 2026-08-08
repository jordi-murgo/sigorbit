# Dataset licence and provenance audit

**Audit date:** 2026-08-07  
**Scope:** the lineage of `sigorbit-c8-257-v1`, not unrelated detector or legacy
encoder experiments.  
**Status:** **source code may remain MIT; public redistribution of the trained
checkpoint is blocked pending upstream permission/legal approval.**

This is an engineering provenance review, not legal advice.

## Executive conclusion

The deployed encoder was trained only from
`rakshitdabral/Signature-Verification-Dataset` at Hugging Face revision
`485e3b6f95ef93a9994b93459933770f69a2e554`. Its card declares MIT, but its own
licence section also says users must comply with original source-dataset terms.

A complete pixel-set comparison establishes that its 7,560 images are a renamed,
PNG-packaged selection of the **genuine** samples from exactly two upstream
benchmarks:

1. all 1,320 genuine CEDAR images: 55 signers × 24;
2. all 6,240 genuine BHSig260 images: 160 Hindi + 100 Bengali signers × 24.

Neither upstream archive contains an open licence grant. The BHSig260 authors
state only that it is available **“for research purposes”**; no commercial or
MIT-compatible redistribution grant was found. Public download, a research
citation, or a third-party `license: mit` label is not evidence that the uploader
owns the right to relicense the underlying biometric images. Therefore:

- do not redistribute either dataset or sample images;
- do not describe the upstream data as MIT;
- do not publish the trained checkpoint under MIT or use it commercially yet;
- obtain written permission covering model training, commercial use and public
  redistribution of derived weights, or retrain on data with verified rights.

Whether model weights are legally derivative of training images varies by facts
and jurisdiction. The absence of clear upstream terms makes an unqualified MIT
weight release an avoidable risk.

## What actually entered the model lineage

| Stage | Gradient-training data | Selection/evaluation data | Other data |
|---|---|---|---|
| C8 backbone from random init | HF `train`, 6,000 genuine images / 250 identities | HF `validation`, 768 / 32 | synthetic rotations of the same images |
| Rotation-augmented C8 checkpoint | same HF `train` split | same HF `validation` split | no external pretrained backbone |
| 257px angle-only canonicalizer pretraining | same HF `train` split | none for gradients | synthetic known-angle rotations |
| 257px joint ArcFace/orientation/consistency training | same HF `train` split | HF `validation` for checkpoint selection | synthetic rotations; no reflections |
| Final test/calibration | no gradients | HF `test`, 792 / 33 | nine private Jordi crops and catalogue images for diagnostics only |

The canonicalized lineage does **not** inherit HairyPotato/EfficientNet weights
and does not use YOLO or detector datasets. Private Jordi signatures were not
used to update weights. The 768-image validation split affected checkpoint
selection, and the 792-image test split was later used for A/B evaluation and
threshold calibration; neither supplied gradients, but the test split is no
longer untouched for future claims.

### Reproducibility evidence limit

The final checkpoint and cloud log did not embed/archive the historical command
or dataset revision. The strongest local evidence is the materialized dataset,
its Hugging Face cache metadata, the exact 6,000/768 counts in the training log,
and the recorded cloud workflow that uploaded that local dataset after an HF 429.
All point to revision `485e3b6f95ef93a9994b93459933770f69a2e554`; this is strong operational
evidence, not a cryptographic binding between dataset bytes and checkpoint.
Local split fingerprints are train `b605607039e924e8`, validation
`3453e783289950ed`, and test `58f6ea1b5d6ba4a2`.

### Exact language/source composition

The aggregate uses contiguous identities rather than a language-stratified
split:

| Aggregate labels | Source | Split contribution |
|---:|---|---|
| 0–54 | CEDAR English/Latin, 55 identities | all in train: 1,320 images |
| 55–214 | BHSig260 Hindi, 160 identities | all in train: 3,840 images |
| 215–249 | BHSig260 Bengali, first 35 identities | train: 840 images |
| 250–281 | BHSig260 Bengali, next 32 identities | validation: 768 images |
| 282–314 | BHSig260 Bengali, final 33 identities | test: 792 images |

Consequently, the published validation and test scores are **Bengali-only**;
they are not a balanced multilingual estimate. Training saw CEDAR, Hindi and 35
Bengali identities. No skilled-forgery images from either upstream archive were
included, so identity-retrieval results are not forgery-detection results.

## Evidence and licence findings

### 1. Hugging Face aggregate

- Repository: <https://huggingface.co/datasets/rakshitdabral/Signature-Verification-Dataset>
- Audited revision: <https://huggingface.co/datasets/rakshitdabral/Signature-Verification-Dataset/tree/485e3b6f95ef93a9994b93459933770f69a2e554>
- Revision README: <https://huggingface.co/datasets/rakshitdabral/Signature-Verification-Dataset/blob/485e3b6f95ef93a9994b93459933770f69a2e554/README.md>
- Data-addition commit: `50fdf3d712f68b1884af7b8bbb31fb0e6e6f8dae`
- Card metadata: `license: mit`

The card states that the English portion contains CEDAR and that Hindi/Bengali
were “curated”, without naming BHSig260 or documenting consent/source. Its licence
section says both:

> “This dataset is released under the MIT License.”

and:

> “Please ensure compliance with the licensing terms of any original source
> datasets (such as the CEDAR Signature Dataset) when using or redistributing
> derived data.”

The MIT tag was already present in the initial 21-byte README before the images
were added; no upstream licence texts or provenance manifest were added with the
data. This is an uploader assertion, not a demonstrated chain of rights. Hugging
Face's licence guidance tells users to seek out and respect project licences
(<https://huggingface.co/docs/hub/repositories-licenses>), and its uploader
warranties do not create missing upstream permissions.

### Immediate downstream repack

The file ordering, corpus composition and representative source files match
Ishani Kathuria's Kaggle dataset “Handwritten Signature Datasets”:
<https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets>
(API metadata: <https://www.kaggle.com/api/v1/datasets/view/ishanikathuria/handwritten-signature-datasets>).
Its Kaggle licence field is **“Other (specified in description)”**. The
description lists CEDAR and BHSig260 counts and acknowledgements but supplies no
licence grant. It links the same official CEDAR/BHSig260 sources audited below.
A downstream repack cannot create broader rights than its inputs.

### 2. CEDAR Signature Dataset

- Project page: <https://cedar.buffalo.edu/signature/>
- CEDAR “Published Data Sets” page: <https://cedar.buffalo.edu/NIJ/publications.html>
- University at Buffalo archive used for comparison:
  <https://cedar.buffalo.edu/NIJ/data/signatures.rar>
- Archive bytes: `253587033`
- Archive SHA-256:
  `f74b859352783b82399c1be48078b79ad637160ba11f16baf92911dd5568f4d6`

The original CEDAR paper is available from the University at Buffalo:
<https://cedar.buffalo.edu/~srihari/papers/IJPRAI2004.pdf>. It says Database A
was built at CEDAR from 55 writers with 24 genuine samples and that about 20
other people produced skilled forgeries. This documents collection, not a
licence.

The official archive contains 1,320 genuine and 1,320 forged PNGs. Its only
`Readme.txt` says:

> “There are two folders ... full_forg ... 1320 forgery signatures ... full_org
> ... 1320 genuine signatures.”

No `LICENSE`, `COPYING`, terms of use, commercial-use grant or redistribution
grant is present in the archive. The CEDAR web page describes research and gives
a contact address but does not state an open-data licence. The safest CEDAR status is **no explicit licence found / permission required**,
not MIT.

All 55 sets of 24 genuine images in the HF aggregate matched the official CEDAR
archive pixel-for-pixel after decoding.

### 3. BHSig260

- Dataset paper record and accepted manuscript:
  <https://research-repository.griffith.edu.au/items/98077438-5bd2-5440-bbcd-2df09b6c165b>
- Paper: S. Pal, A. R. Alaei, U. Pal and M. Blumenstein, “Performance of an
  Off-Line Signature Verification Method Based on Texture Features on a Large
  Indic-Script Signature Dataset,” DAS 2016, DOI `10.1109/DAS.2016.48`.
- Original public archive linked by common benchmark listings/Kaggle:
  <https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/view>
- Archive filename/bytes: `BHSig260.zip`, `48500114`
- Archive SHA-256:
  `560ea9a569e0735005ec1ba6bc413efb8770f6076bfa17fc57ba3fa6798301ff`

The archive contains 24 genuine and 30 forged TIFFs for each of 160 Hindi and
100 Bengali signers, plus pair/list files. It contains no licence or consent
statement. The accepted manuscript says: “The BHSig260 dataset introduced in
this research work is publicly available for research purposes.” The paper says the data came from 260 people of different educational
backgrounds and ages, collected in two sessions, and twice describes the dataset
as publicly available **“for research purposes.”** This is affirmative evidence
of a limited research-use scope, not an MIT/open-data or commercial grant. The
IEEE copyright notice covers the paper and is not a separate dataset licence.

The Kaggle mirror “Handwritten Signature Datasets” labels its licence as
“Other (specified in description)”, but the description only gives corpus counts
and acknowledgements; it does not provide a permission grant. Other mirrors marked
“Unknown” or unlicensed cannot improve the upstream rights position.

All 260 sets of 24 genuine images in the HF aggregate matched the BHSig260
archive pixel-for-pixel after decoding. No BHSig260 forgeries were used.

## Pixel-provenance verification

For every local HF image, the audit hashed:

```text
SHA256("<width>x<height>:L:" || decoded_grayscale_pixels)
```

and compared per-identity sets against the official/source archives:

| Source | Identities matched | Images matched | Result |
|---|---:|---:|---|
| CEDAR `full_org` | 55/55 | 1,320/1,320 | exact decoded-pixel sets |
| BHSig260 Hindi genuine | 160/160 | 3,840/3,840 | exact decoded-pixel sets |
| BHSig260 Bengali genuine | 100/100 | 2,400/2,400 | exact decoded-pixel sets |

This establishes corpus identity independently of the incomplete HF card. It
does not itself establish a licence.

## Privacy and biometric rights

The archives contain real handwritten signatures and are used to learn vectors
for identity comparison. Besides copyright/database rights, publication needs a
privacy/biometric review: the available archives/cards do not document participant
consent scope, commercial reuse, model release, retention or withdrawal. An MIT
copyright licence would not waive privacy, personality, data-protection or
biometric rights. In an EU context, GDPR Articles 4(14) and 9 specifically address
biometric data used for unique identification:
<https://eur-lex.europa.eu/eli/reg/2016/679/oj>.

## Required release actions

A reusable request is provided in `PERMISSION_REQUEST_TEMPLATE.md`.

### Ask CEDAR / University at Buffalo

Request written confirmation that permits:

1. training an identity embedding model on `full_org`;
2. commercial and non-commercial use;
3. worldwide public redistribution of trained parameter weights under MIT (or a
   stated alternative model licence);
4. publication of aggregate evaluation results;
5. the required attribution/citation and any use restrictions;
6. confirmation that participant consent covers these uses.

### Ask the BHSig260 rights holder/authors

Request the same rights for the 6,240 genuine samples and identify the entity
that owns the dataset—not merely the IEEE paper. Ask for the original participant
consent scope and a canonical licence file/source URL.

### Ask the Hugging Face uploader

Request a source manifest and evidence of authority to apply MIT to CEDAR and
BHSig260 material. The current card's caveat is not a sublicence chain.

### If permission cannot be obtained

- publish SigOrbit source/architecture/API under MIT **without weights**;
- retrain from random initialization on signatures collected with explicit
  consent and a licence covering public derived-weight release; or
- use a synthetic/consented dataset whose provenance and intended biometric use
  are documented, then rerun the full evaluation and model card.

## Release decision matrix

| Artifact | Current decision |
|---|---|
| SigOrbit original source code | MIT, subject to contributor/company ownership confirmation |
| Training/evaluation images | never redistribute |
| Dataset claim “MIT” | do not repeat as an unqualified upstream licence |
| `sigorbit-c8-257-v1` checkpoint | **hold from public release** |
| Architecture/training description and aggregate metrics | publishable with citations and no personal samples, subject to normal review |
| Private Jordi images/embeddings | never publish |
