# Security review

**Review date:** 2026-08-08  
**Scope:** SigOrbit 0.1.0 source package, checkpoint loader, preprocessing and the
example FastAPI embedding service.  
**Result:** no HTTP path-traversal or attacker-selected filesystem write/read was
found; identified resource-exhaustion, dependency-floor and artifact-integrity
risks were hardened before release. An independent final re-review found no
remaining HIGH or MEDIUM blocker under the documented deployment assumptions.

This review reduces known risk; it is not a guarantee that the software has no
vulnerabilities and is not a substitute for deployment-specific penetration
testing.

## Attack-surface conclusion

`POST /embed` ignores `UploadFile.filename`, never constructs an application
filesystem path from request data, and passes only bounded bytes to Pillow. It
has no checkpoint-upload, URL-fetch, database or shell-command endpoint.
Consequently, names such as `../../etc/passwd` do not provide path traversal in
the supplied API; an adversarial regression test verifies that only bytes reach
the encoder and no named file is created.

The in-process library deliberately supports `SignatureEncoder.embed(str_or_Path)`
for trusted local applications. That API can read the path supplied by its
caller. Applications must never forward an HTTP filename/form/query value into
that trusted-path interface.

## Controls verified

| Area | Control |
|---|---|
| Authentication | optional bearer authentication is evaluated in ASGI middleware before reading the body; CLI refuses unauthenticated non-loopback binding |
| Aggregate backpressure | a full-request semaphore is acquired before multipart parsing; excess work receives 503 |
| Request size | `Content-Length` and streaming/chunked bytes are bounded before multipart parsing |
| File size | endpoint reads at most the configured maximum plus one byte and closes the upload |
| Image decoding | Pillow plugin allowlist is supplied at `Image.open`; PNG/JPEG/WebP only; 4,194,304-pixel cap before full decode |
| Error handling | malformed inputs receive generic errors without decoder paths or exception details |
| Checkpoints | HTTP SHA-256 required; immutable 128 MiB-bounded snapshot; `weights_only=True`; exact release-config allowlist; strict schema/state |
| Runtime | one worker and loopback bind by default; encoder forward pass is locked and uses inference mode |
| Persistence | API does not store uploads or embeddings and has no signer database |
| Packaging | public wheel/sdist contain no `.pt`, datasets, signatures or databases |

## Findings remediated

1. Replaced unbounded `UploadFile.read()` and post-parse authentication with
   pre-body authentication, raw request limiting, bounded file reads and
   full-request concurrency backpressure.
2. Restricted Pillow at plugin selection time and reduced the pixel budget from
   Pillow's permissive default to a signature-crop-specific bound.
3. Raised dependency floors to patched versions: PyTorch ≥2.7, Pillow ≥12.3,
   FastAPI ≥0.141.1, Uvicorn ≥0.52.1 and python-multipart ≥0.0.32.
4. Made the HTTP service require a checkpoint digest, deserializing the exact
   verified immutable bytes with a 128 MiB cap and rejecting excessive model
   configurations before model construction; the default loader accepts only the
   published 257/C8 architecture tuple.
5. Removed internal exception text from client responses.
6. Rebuilt artifacts after hardening and moved the optional API command behind a
   clear `sigorbit[api]` dependency check.
7. Changed CI to the frozen lock and added Bandit, pip-audit and Dependabot.

## Verification performed

- adversarial tests: traversal-style filename, corrupt/empty/oversized upload,
  oversized `Content-Length`, chunked body limit, pre-body authentication,
  request concurrency, unsupported format, pixel bomb, exception redaction,
  digest mismatch, checkpoint byte and architecture limits;
- complete tests on Python 3.10, 3.11, 3.12 and the development Python 3.13;
- real 256-D embedding through an installed wheel, using the hash-pinned local
  checkpoint and bearer authentication;
- Ruff and Bandit static checks;
- pip-audit of the frozen runtime dependency export: no known vulnerabilities;
- wheel/sdist inspection: no checkpoint and hardened source present.

## Residual deployment risks

The example server does not provide TLS, distributed rate limiting, identity
management, secret rotation, tenant isolation or a web-application firewall.
Internet exposure must use a maintained ingress/reverse proxy, TLS,
authentication, connection/header/request timeouts and per-client rate limits.
Run it non-root with read-only filesystems and CPU/RAM/GPU limits. Patch image,
ML and multipart dependencies continuously.

A valid checksum protects integrity only if the expected digest and file
permissions are managed securely. `weights_only=True` is not a sandbox against
all denial-of-service behavior or future PyTorch bugs. Biometric privacy,
dataset permission and model misuse are separate release gates described in
`SECURITY.md`, `DATASET_LICENSE_AUDIT.md` and `RELEASING.md`.
