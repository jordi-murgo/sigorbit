import numpy as np
import pytest
from PIL import Image

from sigorbit.preprocessing import image_to_tensor, open_image


def test_preprocess_range_shape_and_dtype():
    array = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    tensor = image_to_tensor(array, 2)
    assert tensor.shape == (1, 2, 2)
    assert str(tensor.dtype) == "torch.float32"
    assert tensor[0, 0, 0].item() == -1.0
    assert tensor[0, 0, 1].item() == 1.0
    assert tensor[0, 1, 0].item() == pytest.approx(128 / 127.5 - 1, abs=1e-7)


def test_rgb_and_encoded_bytes_are_supported(tmp_path):
    image = Image.new("RGB", (11, 7), "white")
    path = tmp_path / "image.png"
    image.save(path)
    tensor_path = image_to_tensor(path, 17)
    tensor_bytes = image_to_tensor(path.read_bytes(), 17)
    assert tensor_path.shape == (1, 17, 17)
    assert np.array_equal(tensor_path.numpy(), tensor_bytes.numpy())


def test_float_array_requires_unit_range():
    assert open_image(np.zeros((4, 4), dtype=np.float32)).mode == "L"
    with pytest.raises(ValueError, match="must be in"):
        open_image(np.full((4, 4), 2.0, dtype=np.float32))
