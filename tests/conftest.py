from __future__ import annotations

import pytest
import torch

from sigorbit.model import CanonicalizedEncoder, ModelConfig


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("weights") / "tiny.pt"
    config = ModelConfig(
        input_size=33, rotations=4, widths=(1, 1, 1, 1), embedding_dim=8, dropout=0.0
    )
    model = CanonicalizedEncoder(config).eval()
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.endswith((".filter", ".expanded_bias"))
    }
    torch.save(
        {
            "format_version": 1,
            "config": config.to_checkpoint_dict(),
            "model_state_dict": state,
            "metadata": {
                "model_id": "tiny-test-v1",
                "preprocess_version": "tiny-gray-33-v1",
            },
        },
        path,
    )
    return path
