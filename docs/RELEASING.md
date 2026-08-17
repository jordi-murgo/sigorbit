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
- [ ] obtain/review written CEDAR and BHSig260 permission for training,
      commercial use and public derived-weight redistribution using
      `PERMISSION_REQUEST_TEMPLATE.md`;
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

## PyPI publishing

Trusted Publisher coordinates:

- project: `sigorbit`;
- owner/repository: `jordi-murgo/sigorbit`;
- workflow: `.github/workflows/publish.yml`;
- GitHub environment: `pypi`.

Publishing is triggered only by a GitHub release. The workflow verifies that the
`vX.Y.Z` tag equals `pyproject.toml`, rebuilds in isolation, rejects model/data
artifacts, runs `twine check`, and publishes via OIDC without a long-lived PyPI
token.

## Technical gates

- [x] `ruff check .` and non-integration tests pass on Python 3.10–3.13;
- [x] build wheel and sdist, inspect them and smoke the installed wheel;
- [x] scan the worktree/history/artifacts for credential patterns; detected
      high-entropy values were documented SHA-256 hashes/fingerprints only;
- [x] verify local checkpoint SHA-256 against the model card;
- [x] verify slim-versus-source embedding parity;
- [ ] complete the pending supported NVIDIA GPU smoke (CPU smoke passed);
- [x] test FastAPI health/embed/auth, malformed/traversal/oversize/chunked inputs;
- [x] run the source security review, Bandit and frozen-lock pip-audit;
- [ ] tag a signed release and publish checksums.
