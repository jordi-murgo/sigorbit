"""Safe checkpoint loading and model identity helpers."""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import torch

from .model import CanonicalizedEncoder, ModelConfig

DERIVED_BUFFER_SUFFIXES = (".filter", ".expanded_bias")
DEFAULT_MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_MODEL_DENSE_UPPER_BOUND = 250_000_000
DEFAULT_MAX_MODEL_INPUT_SIZE = 513
APPROVED_RELEASE_CONFIGS = frozenset(
    {
        (257, 8, (24, 48, 96, 128), 256),
    }
)


@dataclass(frozen=True)
class CheckpointInfo:
    """Runtime identity of a loaded checkpoint."""

    path: str
    sha256: str
    model_id: str
    preprocess_version: str
    config: ModelConfig
    metadata: dict[str, Any]


def bundled_checkpoint() -> Path:
    """Return an approved packaged checkpoint, if a distribution includes one.

    The code-only public release intentionally excludes weights until upstream
    permissions are resolved. Deployments should pass an explicit local path.
    """
    candidate = resources.files("sigorbit").joinpath("weights/sigorbit-c8-257-v1.pt")
    path = Path(os.fspath(candidate))
    if not path.is_file():
        raise FileNotFoundError(
            "no approved bundled checkpoint; pass checkpoint= explicitly or set "
            "SIGORBIT_CHECKPOINT for the API"
        )
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_identity(config: ModelConfig) -> tuple[int, int, tuple[int, ...], int]:
    return (config.input_size, config.rotations, config.widths, config.embedding_dim)


def _dense_parameter_upper_bound(config: ModelConfig) -> int:
    """Conservative complexity guard before constructing an e2cnn model."""
    channels = [width * config.rotations for width in config.widths]
    estimate = channels[0] * 7 * 7
    previous = channels[0]
    for output in channels[1:]:
        estimate += (previous * output + output * output) * 5 * 5
        previous = output
    estimate += config.widths[-1] * 512 + 512 * config.embedding_dim
    return estimate


def load_model(
    checkpoint: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
    expected_sha256: str | None = None,
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    strict_release_config: bool = True,
) -> tuple[CanonicalizedEncoder, CheckpointInfo]:
    """Load a SigOrbit checkpoint without enabling arbitrary pickle objects.

    Checkpoints are tensor/state-dict packages and are read with
    ``weights_only=True``. Missing e2cnn ``filter`` buffers are expected in the
    slim release checkpoint because they are derived from trained weights.
    The default accepts only the published SigOrbit architecture. Disabling
    ``strict_release_config`` is for trusted research checkpoints only; bounded
    size and complexity guards still apply.
    """
    path = Path(checkpoint) if checkpoint is not None else bundled_checkpoint()
    if not path.is_file():
        raise FileNotFoundError(f"SigOrbit checkpoint not found: {path}")
    if max_checkpoint_bytes < 1:
        raise ValueError("max_checkpoint_bytes must be positive")
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.lower()
        if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters")

    # Read a bounded immutable snapshot, then hash and deserialize those exact
    # bytes. This prevents path replacement, in-place mutation races, and growth
    # beyond the size limit between integrity verification and deserialization.
    with path.open("rb") as source:
        if os.fstat(source.fileno()).st_size > max_checkpoint_bytes:
            raise ValueError("checkpoint exceeds the configured byte limit")
        payload = source.read(max_checkpoint_bytes + 1)
    if len(payload) > max_checkpoint_bytes:
        raise ValueError("checkpoint exceeds the configured byte limit")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("checkpoint SHA-256 does not match the expected value")
    package = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(package, dict):
        raise ValueError("invalid checkpoint: expected a dictionary")
    if package.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format_version")
    raw_config = package.get("config")
    state = package.get("model_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(state, dict):
        raise ValueError("invalid checkpoint: missing config or model_state_dict")

    config = ModelConfig.from_checkpoint_dict(raw_config)
    if strict_release_config and _config_identity(config) not in APPROVED_RELEASE_CONFIGS:
        raise ValueError("checkpoint configuration is not an approved SigOrbit release")
    if config.input_size > DEFAULT_MAX_MODEL_INPUT_SIZE:
        raise ValueError("checkpoint input_size exceeds the safe loader limit")
    if _dense_parameter_upper_bound(config) > DEFAULT_MAX_MODEL_DENSE_UPPER_BOUND:
        raise ValueError("checkpoint architecture exceeds the safe complexity limit")
    model = CanonicalizedEncoder(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_derived_missing = [key for key in missing if not key.endswith(DERIVED_BUFFER_SUFFIXES)]
    if non_derived_missing or unexpected:
        raise ValueError(
            f"incompatible checkpoint; missing={non_derived_missing}, unexpected={list(unexpected)}"
        )
    model = model.to(device).eval()

    raw_metadata = package.get("metadata") or {}
    if not isinstance(raw_metadata, dict):
        raise ValueError("invalid checkpoint: metadata must be a dictionary")
    metadata = dict(raw_metadata)
    model_id = str(metadata.get("model_id") or f"sha256:{digest[:16]}")
    preprocess_version = str(
        metadata.get("preprocess_version") or f"sigorbit-gray-square-{config.input_size}-v1"
    )
    info = CheckpointInfo(
        path=str(path),
        sha256=digest,
        model_id=model_id,
        preprocess_version=preprocess_version,
        config=config,
        metadata=metadata,
    )
    return model, info
