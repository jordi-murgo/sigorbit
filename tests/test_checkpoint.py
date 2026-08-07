from pathlib import Path

import pytest
import torch

from sigorbit.checkpoint import file_sha256, load_model


@pytest.mark.integration
def test_local_release_checkpoint_checksum_and_metadata():
    import os

    path_value = os.getenv("SIGORBIT_TEST_CHECKPOINT")
    if not path_value:
        pytest.skip("set SIGORBIT_TEST_CHECKPOINT to the approved/local artifact")
    path = Path(path_value)
    assert file_sha256(path) == "ff62889e2c5172762e894f4ee05162e0109d06c863b1fa83a7bcc6baf1dc0963"
    package = torch.load(path, map_location="cpu", weights_only=True)
    assert package["format_version"] == 1
    assert package["metadata"]["model_id"] == "sigorbit-c8-257-v1"
    assert package["metadata"]["weights_release_status"].startswith("withheld")
    assert not any(
        key.endswith((".filter", ".expanded_bias")) for key in package["model_state_dict"]
    )


def test_loader_rejects_noncanonical_checkpoint(tiny_checkpoint, tmp_path):
    package = torch.load(tiny_checkpoint, map_location="cpu", weights_only=True)
    package["config"]["canonicalized"] = False
    path = tmp_path / "bad.pt"
    torch.save(package, path)
    with pytest.raises(ValueError, match="not a canonicalized"):
        load_model(path)


def test_loader_rejects_missing_learned_key(tiny_checkpoint, tmp_path):
    package = torch.load(tiny_checkpoint, map_location="cpu", weights_only=True)
    learned_key = next(key for key in package["model_state_dict"] if key.endswith("weight"))
    del package["model_state_dict"][learned_key]
    path = tmp_path / "bad.pt"
    torch.save(package, path)
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        load_model(path)
