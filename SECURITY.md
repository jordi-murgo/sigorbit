# Security and biometric-data handling

Report security issues privately to the repository maintainers before opening a
public issue. A public security contact will be added before the first hosted
release.

SigOrbit processes biometric-like data. Applications should:

- reject oversized or malformed uploads and run image decoders in patched
  environments;
- avoid logging uploads, embeddings or identity decisions;
- encrypt data in transit and at rest and isolate tenant indexes;
- bind every vector to model and preprocessing identities;
- define retention, deletion, access-control and incident-response procedures;
- consider template-protection/cancelable-biometric techniques;
- calibrate for unknown identities and skilled forgeries in the target domain;
- keep a human review path for consequential decisions.

PyTorch `.pt` artifacts are loaded with `weights_only=True`; do not weaken this
for untrusted files. Verify release artifact checksums before deployment.
