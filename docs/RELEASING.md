# Release checklist

The repository is initialized for an MIT-licensed source release, but pushing it
publicly should be a deliberate review, especially because signatures are
biometric data.

## Naming

- [x] `sigorbit` absent from PyPI exact-name lookup on 2026-08-07.
- [x] no exact GitHub repository result in the same point-in-time check.
- [ ] perform final trademark/domain review immediately before publication.

## Source and dependency rights

- [ ] confirm the company/contributors own or may relicense every extracted
      source file under MIT;
- [ ] retain the e2cnn BSD 3-Clause Clear notice, including its patent clause;
- [ ] generate an SBOM and license report from the actual lock/container;
- [ ] verify no HairyPotato GPL code or weights, YOLO/Ultralytics artifacts,
      training images, private signatures, databases or logs enter the release.

## Dataset and weights

The completed `DATASET_LICENSE_AUDIT.md` matched all 7,560 decoded images to
1,320 genuine CEDAR and 6,240 genuine BHSig260 samples. CEDAR has no explicit
licence grant; BHSig260 is stated only to be available “for research purposes”. The aggregate Hub card declares MIT
but explicitly defers to upstream terms; that tag is not a demonstrated right to
relicense the images. Before publishing weights:

- [x] pin the exact Hub revision and card (`485e3b6f95ef93a9994b93459933770f69a2e554`);
- [ ] obtain/review written CEDAR and BHSig260 permission for training, commercial use and public derived-weight redistribution;
- [ ] document provenance and consent for every included language/corpus;
- [ ] decide and record the weight license in a dedicated file;
- [ ] approve biometric/privacy and model-inversion risk;
- [ ] cryptographically attest the lineage: random-init C8 → rotation-augmented
      C8 → 257px canonicalized model;
- [ ] ensure no sample image from the training or private evaluation sets is
      present in tests, docs, wheels or source distributions.

Until these boxes are approved, the release decision is **code-only**. Treat
the local checkpoint as a pre-release engineering artifact, not a public
redistribution grant; exclude it from any public remote, wheel, sdist or release.

## Technical gates

- [ ] `ruff check .` and `pytest` pass in a clean environment;
- [ ] build wheel and sdist, install wheel into a clean environment;
- [ ] scan archive contents and secrets;
- [ ] verify bundled/checkpoint SHA-256 against the model card;
- [ ] verify slim-versus-source embedding parity;
- [ ] CPU smoke and at least one supported NVIDIA GPU smoke;
- [ ] FastAPI `/health`, valid `/embed`, corrupt upload and oversize tests;
- [ ] tag a signed release and publish checksums.
