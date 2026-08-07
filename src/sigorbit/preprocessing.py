"""The versioned image-to-tensor contract used by SigOrbit."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from PIL import Image

ImageInput: TypeAlias = Image.Image | np.ndarray | bytes | bytearray | str | Path


def open_image(value: ImageInput) -> Image.Image:
    """Decode a supported image input and fully detach it from its source."""
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            image.load()
            return image.copy()
    if isinstance(value, (bytes, bytearray)):
        with Image.open(io.BytesIO(value)) as image:
            image.load()
            return image.copy()
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size:
                if float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
                    array = np.rint(array * 255.0).astype(np.uint8)
                else:
                    raise ValueError("floating image arrays must be in [0, 1]")
            else:
                raise ValueError("image arrays must use uint8 or floats in [0, 1]")
        return Image.fromarray(array)
    raise TypeError(f"unsupported image input: {type(value).__name__}")


def image_to_tensor(value: ImageInput, input_size: int) -> torch.Tensor:
    """Convert an image to ``(1,H,W)`` grayscale in the model's ``[-1,1]`` range."""
    image = (
        open_image(value).convert("L").resize((input_size, input_size), Image.Resampling.BICUBIC)
    )
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array.copy()).unsqueeze(0)
    return tensor.div_(127.5).sub_(1.0)
