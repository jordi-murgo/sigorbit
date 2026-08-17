# Permission request template

Use a separate request for CEDAR/University at Buffalo and for the BHSig260
rightsholder/authors. Replace brackets and keep the written reply with release
records.

## Subject

Permission request: ML training and public derived-weight release from
`[CEDAR Signature Dataset / BHSig260]`

## Body

Dear `[dataset owner/rightsholder]`,

We are preparing SigOrbit, an open-source Python implementation of a
rotation-robust handwritten-signature embedding model. The source code is
already published at https://github.com/jordi-murgo/sigorbit and distributed
on PyPI at https://pypi.org/project/sigorbit/ . We intend to distribute the
trained model weights as part of the same PyPI package, so that users can
install the library and run inference without a separate download.
We identified the exact training subset as `[1,320 CEDAR genuine images / 6,240 BHSig260 genuine images]`.
We do not plan to redistribute any source signature image.

Please confirm whether you authorize all of the following:

1. copying and preprocessing the genuine images for machine-learning training;
2. non-commercial and commercial research/deployment;
3. publishing the trained neural-network parameters worldwide;
4. licensing those derived weights under `[proposed licence]`, including rights
   to use, modify, redistribute and sublicense them;
5. publishing aggregate benchmark results and corpus citations;
6. retaining the minimum secured training copy required for reproducibility.

Please state required attribution, field-of-use restrictions, fees, security or
deletion obligations, and whether participant consent/releases cover biometric
model training, commercial use and public derived-weight publication. Please
also identify the entity that owns or controls the relevant dataset rights.

Model details:

- architecture: SO(2) canonicalizer + C8 steerable CNN;
- output: 256-D identity embedding;
- training used genuine samples only, with synthetic rotations;
- no raw images or reconstructed samples will be shipped;
- the checkpoint is currently withheld pending your response.
- source code: https://github.com/jordi-murgo/sigorbit
- PyPI package: https://pypi.org/project/sigorbit/

A reply should be reviewed by our legal/privacy team before release.

Sincerely,
`[name, company, contact]`
