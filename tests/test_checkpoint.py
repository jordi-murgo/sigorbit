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
    assert file_sha256(path) == "ec8d99f887f5a2658d93b14a14911b29a1411e9cf142efa85862a47b30cd233e"
    package = torch.load(path, map_location="cpu", weights_only=True)
    _, info = load_model(path, expected_sha256=file_sha256(path))
    assert info.model_id == "sigorbit-c8-257-v1"
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
        load_model(path, strict_release_config=False)


def test_loader_rejects_missing_learned_key(tiny_checkpoint, tmp_path):
    package = torch.load(tiny_checkpoint, map_location="cpu", weights_only=True)
    learned_key = next(key for key in package["model_state_dict"] if key.endswith("weight"))
    del package["model_state_dict"][learned_key]
    path = tmp_path / "bad.pt"
    torch.save(package, path)
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        load_model(path, strict_release_config=False)


def test_loader_verifies_sha256_before_deserialization(tiny_checkpoint):
    digest = file_sha256(tiny_checkpoint)
    model, info = load_model(tiny_checkpoint, expected_sha256=digest, strict_release_config=False)
    assert model is not None
    assert info.sha256 == digest
    with pytest.raises(ValueError, match="does not match"):
        load_model(tiny_checkpoint, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="64 hexadecimal"):
        load_model(tiny_checkpoint, expected_sha256="not-a-sha")


def test_loader_enforces_checkpoint_byte_limit(tiny_checkpoint):
    with pytest.raises(ValueError, match="byte limit"):
        load_model(tiny_checkpoint, max_checkpoint_bytes=1)


def test_loader_rejects_excessive_model_configuration(tiny_checkpoint, tmp_path):
    package = torch.load(tiny_checkpoint, map_location="cpu", weights_only=True)
    package["config"]["N"] = 32
    package["config"]["widths"] = [1024, 1024, 1024, 1024]
    path = tmp_path / "oversized-architecture.pt"
    torch.save(package, path)
    with pytest.raises(ValueError, match="complexity limit"):
        load_model(path, strict_release_config=False)


def test_loader_defaults_to_approved_release_configurations(tiny_checkpoint):
    with pytest.raises(ValueError, match="approved SigOrbit release"):
        load_model(tiny_checkpoint)
