import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from sigorbit.api import create_app
from sigorbit.encoder import EmbeddingResult
from sigorbit.preprocessing import open_image


class FakeEncoder:
    model_id = "fake-v1"
    preprocess_version = "fake-gray-v1"
    input_size = 17
    embedding_dim = 8
    device = "cpu"

    def embed_with_details(self, data):
        open_image(data)
        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        vector[0] = 1.0
        return EmbeddingResult(vector, 12.5)


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (12, 5), "white").save(stream, format="PNG")
    return stream.getvalue()


def test_health_and_embed():
    app = create_app(encoder=FakeEncoder())
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_id"] == "fake-v1"
        response = client.post(
            "/embed", files={"file": ("signature.png", png_bytes(), "image/png")}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["dimensions"] == 8
    assert body["normalized"] is True
    assert body["orientation_degrees"] == 12.5
    assert len(body["embedding"]) == 8


def test_corrupt_empty_and_oversized_uploads(monkeypatch):
    app = create_app(encoder=FakeEncoder())
    with TestClient(app) as client:
        corrupt = client.post("/embed", files={"file": ("bad.png", b"not an image", "image/png")})
        empty = client.post("/embed", files={"file": ("empty.png", b"", "image/png")})
        monkeypatch.setenv("SIGORBIT_MAX_UPLOAD_BYTES", "2")
        large = client.post("/embed", files={"file": ("large.png", png_bytes(), "image/png")})
    assert corrupt.status_code == 400
    assert empty.status_code == 400
    assert large.status_code == 413
