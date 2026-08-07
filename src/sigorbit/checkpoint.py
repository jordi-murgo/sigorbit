"""Safe checkpoint loading and model identity helpers."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import torch

from .model import CanonicalizedEncoder, ModelConfig

DERIVED_BUFFER_SUFFIXES = (".filter", ".expanded_bias")


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


def load_model(
    checkpoint: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CanonicalizedEncoder, CheckpointInfo]:
    """Load a SigOrbit checkpoint without enabling arbitrary pickle objects.

    Checkpoints are tensor/state-dict packages and are read with
    ``weights_only=True``. Missing e2cnn ``filter`` buffers are expected in the
    slim release checkpoint because they are derived from trained weights.
    """
    path = Path(checkpoint) if checkpoint is not None else bundled_checkpoint()
    if not path.is_file():
        raise FileNotFoundError(f"SigOrbit checkpoint not found: {path}")
    package = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(package, dict):
        raise ValueError("invalid checkpoint: expected a dictionary")
    if package.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format_version")
    raw_config = package.get("config")
    state = package.get("model_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(state, dict):
        raise ValueError("invalid checkpoint: missing config or model_state_dict")

    config = ModelConfig.from_checkpoint_dict(raw_config)
    model = CanonicalizedEncoder(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_derived_missing = [key for key in missing if not key.endswith(DERIVED_BUFFER_SUFFIXES)]
    if non_derived_missing or unexpected:
        raise ValueError(
            f"incompatible checkpoint; missing={non_derived_missing}, unexpected={list(unexpected)}"
        )
    model = model.to(device).eval()

    digest = file_sha256(path)
    metadata = dict(package.get("metadata") or {})
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
