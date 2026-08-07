"""High-level Python API for signature embeddings."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from .checkpoint import CheckpointInfo, load_model
from .model import CanonicalizedEncoder
from .preprocessing import ImageInput, image_to_tensor


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding plus the orientation predicted by the canonicalizer."""

    embedding: np.ndarray
    orientation_degrees: float


def resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm was requested but torch.cuda.is_available() is false")
    return resolved


class SignatureEncoder:
    """Thread-safe, checkpoint-aware signature embedding interface.

    Inputs are signature crops, not complete documents. Every returned vector is
    a float32, L2-normalized NumPy array. Embeddings from different ``model_id``
    or ``preprocess_version`` values must never be mixed in one index.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str | torch.device = "auto",
    ):
        self.device = resolve_device(device)
        model, info = load_model(checkpoint, device=self.device)
        self.model: CanonicalizedEncoder = model
        self.info: CheckpointInfo = info
        self._lock = threading.Lock()

    @property
    def input_size(self) -> int:
        return self.info.config.input_size

    @property
    def embedding_dim(self) -> int:
        return self.info.config.embedding_dim

    @property
    def model_id(self) -> str:
        return self.info.model_id

    @property
    def preprocess_version(self) -> str:
        return self.info.preprocess_version

    def _forward(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        batch = batch.to(self.device)
        with self._lock, torch.inference_mode():
            output = self.model(batch, return_orientation=True)
            embeddings, cos_sin = cast(tuple[torch.Tensor, torch.Tensor], output)
        return (
            embeddings.detach().cpu().numpy().astype(np.float32, copy=False),
            cos_sin.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    def embed(self, image: ImageInput) -> np.ndarray:
        """Generate one 256-dimensional L2-normalized embedding."""
        tensor = image_to_tensor(image, self.input_size).unsqueeze(0)
        embeddings, _ = self._forward(tensor)
        return embeddings[0]

    def embed_with_details(self, image: ImageInput) -> EmbeddingResult:
        tensor = image_to_tensor(image, self.input_size).unsqueeze(0)
        embeddings, cos_sin = self._forward(tensor)
        angle = math.degrees(math.atan2(float(cos_sin[0, 1]), float(cos_sin[0, 0])))
        return EmbeddingResult(embeddings[0], angle)

    def embed_batch(self, images: Sequence[ImageInput], *, batch_size: int = 16) -> np.ndarray:
        """Embed images in bounded batches while preserving input order."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            tensors = [
                image_to_tensor(image, self.input_size)
                for image in images[start : start + batch_size]
            ]
            embeddings, _ = self._forward(torch.stack(tensors))
            chunks.append(embeddings)
        return np.concatenate(chunks, axis=0)
