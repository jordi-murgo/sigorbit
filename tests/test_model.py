import numpy as np
import torch

from sigorbit import SignatureEncoder
from sigorbit.model import CanonicalizedEncoder, ModelConfig, OrientationCanonicalizer


def test_canonicalizer_starts_at_identity():
    module = OrientationCanonicalizer().eval()
    tensor = torch.randn(2, 1, 33, 33)
    with torch.inference_mode():
        canonical, cos_sin = module(tensor)
    assert canonical.shape == tensor.shape
    assert torch.allclose(cos_sin, torch.tensor([[1.0, 0.0], [1.0, 0.0]]), atol=2e-6)
    assert torch.allclose(cos_sin.norm(dim=1), torch.ones(2), atol=2e-6)
    assert torch.isfinite(canonical).all()


def test_tiny_model_returns_unit_embeddings():
    config = ModelConfig(33, 4, (1, 1, 1, 1), 8, 0.0)
    model = CanonicalizedEncoder(config).eval()
    with torch.inference_mode():
        embeddings = model(torch.randn(2, 1, 33, 33))
    assert embeddings.shape == (2, 8)
    assert torch.isfinite(embeddings).all()
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)


def test_public_encoder_single_batch_and_metadata(tiny_checkpoint):
    encoder = SignatureEncoder(tiny_checkpoint, device="cpu")
    image = np.full((25, 40), 255, dtype=np.uint8)
    first = encoder.embed(image)
    second = encoder.embed(image)
    batch = encoder.embed_batch([image, image], batch_size=2)
    assert first.shape == (8,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.allclose(batch[0], first, atol=1e-6)
    assert np.allclose(batch[1], first, atol=1e-6)
    assert np.isclose(np.linalg.norm(first), 1.0, atol=1e-6)
    assert encoder.model_id == "tiny-test-v1"
    assert encoder.preprocess_version == "tiny-gray-33-v1"
