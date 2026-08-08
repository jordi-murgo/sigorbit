"""The versioned image-to-tensor contract used by SigOrbit."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from PIL import Image

ImageInput: TypeAlias = Image.Image | np.ndarray | bytes | bytearray | str | Path

DEFAULT_MAX_IMAGE_PIXELS = 4_194_304
ALLOWED_ENCODED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


def _validate_dimensions(width: int, height: int, max_pixels: int) -> None:
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if width * height > max_pixels:
        raise ValueError(f"image exceeds the {max_pixels}-pixel safety limit")


def _load_encoded_image(source: str | Path | io.BytesIO, max_pixels: int) -> Image.Image:
    try:
        # Supplying ``formats`` prevents Pillow from invoking unrelated format
        # plugins merely to reject them after header parsing.
        with Image.open(source, formats=tuple(ALLOWED_ENCODED_FORMATS)) as image:
            _validate_dimensions(image.width, image.height, max_pixels)
            image.load()
            return image.copy()
    except Image.DecompressionBombError as exc:
        raise ValueError("image exceeds the pixel safety limit") from exc


def open_image(value: ImageInput, *, max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS) -> Image.Image:
    """Decode a bounded PNG/JPEG/WebP image and detach it from its source.

    ``str`` and ``Path`` values are intentionally supported by the trusted
    in-process library API. The HTTP API never passes an uploaded filename here;
    it supplies the uploaded bytes instead.
    """
    if isinstance(value, Image.Image):
        _validate_dimensions(value.width, value.height, max_pixels)
        return value.copy()
    if isinstance(value, (str, Path)):
        return _load_encoded_image(value, max_pixels)
    if isinstance(value, (bytes, bytearray)):
        return _load_encoded_image(io.BytesIO(value), max_pixels)
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim not in (2, 3):
            raise ValueError("image arrays must have two or three dimensions")
        _validate_dimensions(int(array.shape[1]), int(array.shape[0]), max_pixels)
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


def image_to_tensor(
    value: ImageInput,
    input_size: int,
    *,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> torch.Tensor:
    """Convert an image to ``(1,H,W)`` grayscale in the model's ``[-1,1]`` range."""
    image = (
        open_image(value, max_pixels=max_pixels)
        .convert("L")
        .resize((input_size, input_size), Image.Resampling.BICUBIC)
    )
    array = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array.copy()).unsqueeze(0)
    return tensor.div_(127.5).sub_(1.0)
