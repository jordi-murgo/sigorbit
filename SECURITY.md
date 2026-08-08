# Security and biometric-data handling

The source-level review and validation record is in
[`docs/SECURITY_REVIEW.md`](docs/SECURITY_REVIEW.md).

Report security issues privately to Jordi Murgó at
[jordi.murgo@gmail.com](mailto:jordi.murgo@gmail.com) or
[jordi.murgo@gft.com](mailto:jordi.murgo@gft.com) before opening a public issue.
Do not attach signatures, embeddings or other personal data to a public report.

## Threat model and implemented controls

The example HTTP API accepts an uploaded cropped image, decodes it and returns an
embedding. It does not accept a client filesystem path, does not use the upload
filename and does not write uploads to application-controlled paths. Therefore
`../../...` upload filenames cannot cause path traversal in the supplied API.
Regression tests exercise this property.

Implemented controls include:

- a raw ASGI request-body limit enforced before multipart parsing, including for
  bodies without `Content-Length`;
- a second bounded read of the uploaded file;
- a 4,194,304-pixel decompression-bomb limit before full image decoding;
- an allowlist of PNG, JPEG and WebP encoded formats;
- bounded concurrent inference with a queue timeout;
- generic client errors that do not expose decoder/checkpoint paths;
- optional constant-time bearer-token comparison through `SIGORBIT_API_KEY`;
- loopback-only binding by default;
- `torch.load(..., weights_only=True)`, a checkpoint byte/complexity limit,
  an exact approved-release configuration allowlist plus state validation, and
  pre-load SHA-256 verification
  required by the HTTP API (optional for trusted in-process callers).

The Python library intentionally accepts `str` and `Path` image inputs for
trusted local callers. This is an arbitrary local file-read capability by design,
not an HTTP feature. Never pass a remote filename, URL parameter or form field to
that interface; pass the received bytes instead.

## Deployment requirements

The example is not a complete Internet-facing security perimeter. If it is
exposed outside localhost:

1. require `SIGORBIT_API_KEY` or replace it with the organization's authenticated
   gateway;
2. terminate TLS at a maintained reverse proxy;
3. enforce IP/client rate limits, request and header timeouts, connection limits,
   and a body limit at that proxy as well;
4. run as a non-root user in a read-only, resource-limited container/process with
   no access to unrelated files;
5. set `SIGORBIT_CHECKPOINT_SHA256` and keep the checkpoint directory read-only;
6. keep Pillow, PyTorch, FastAPI/Starlette and the multipart parser patched;
7. disable or restrict interactive OpenAPI documentation if the deployment
   considers its schema sensitive;
8. monitor only operational metadata—never upload bodies or embeddings.

`weights_only=True` prevents normal pickle global execution but is not a sandbox
against every resource-exhaustion or future dependency bug. Only load an approved,
hash-pinned artifact. The built-in concurrency and size controls reduce resource
exhaustion; distributed denial of service still requires infrastructure controls.

## Biometric handling

SigOrbit processes biometric-like data. Applications should:

- avoid logging uploads, embeddings or identity decisions;
- encrypt data in transit and at rest and isolate tenant indexes;
- bind every vector to model and preprocessing identities;
- define retention, deletion, access-control and incident-response procedures;
- consider template-protection/cancelable-biometric techniques;
- calibrate for unknown identities and skilled forgeries in the target domain;
- keep a human review path for consequential decisions.

Verify release artifact checksums before deployment. See the dataset licence audit
for the separate checkpoint redistribution and consent restrictions.
