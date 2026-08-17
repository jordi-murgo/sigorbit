"""SigOrbit: rotation-robust handwritten-signature embeddings."""

from .checkpoint import CheckpointInfo, bundled_checkpoint, load_model
from .encoder import EmbeddingResult, SignatureEncoder
from .model import CanonicalizedEncoder, ModelConfig, OrientationCanonicalizer, SteerableEncoder

__all__ = [
    "CanonicalizedEncoder",
    "CheckpointInfo",
    "EmbeddingResult",
    "ModelConfig",
    "OrientationCanonicalizer",
    "SignatureEncoder",
    "SteerableEncoder",
    "bundled_checkpoint",
    "load_model",
]
__version__ = "0.2.4"
