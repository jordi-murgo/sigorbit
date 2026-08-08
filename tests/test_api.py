import asyncio
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sigorbit import api as api_module
from sigorbit.api import (
    RequestBodyLimitMiddleware,
    RequestConcurrencyMiddleware,
    create_app,
)
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


class RecordingEncoder(FakeEncoder):
    received = None

    def embed_with_details(self, data):
        self.received = data
        return super().embed_with_details(data)


class LeakyEncoder(FakeEncoder):
    def embed_with_details(self, data):
        raise ValueError("sensitive/internal/path/checkpoint.pt")


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


def test_corrupt_and_empty_uploads_do_not_leak_decoder_details():
    app = create_app(encoder=FakeEncoder())
    with TestClient(app) as client:
        corrupt = client.post("/embed", files={"file": ("bad.png", b"not an image", "image/png")})
        empty = client.post("/embed", files={"file": ("empty.png", b"", "image/png")})
    assert corrupt.status_code == 400
    assert corrupt.json() == {"detail": "invalid or unsupported image"}
    assert empty.status_code == 400


def test_oversized_file_and_request_are_rejected():
    file_limited = create_app(encoder=FakeEncoder(), max_upload_bytes=2, max_request_bytes=1024)
    request_limited = create_app(encoder=FakeEncoder(), max_upload_bytes=10, max_request_bytes=20)
    with TestClient(file_limited) as client:
        file_response = client.post(
            "/embed", files={"file": ("large.png", png_bytes(), "image/png")}
        )
    with TestClient(request_limited) as client:
        request_response = client.post(
            "/embed", content=b"x" * 21, headers={"content-type": "application/octet-stream"}
        )
    assert file_response.status_code == 413
    assert request_response.status_code == 413


def test_bodyless_routes_and_invalid_content_type_cannot_bypass_raw_limit():
    app = create_app(encoder=FakeEncoder(), max_upload_bytes=10, max_request_bytes=20)

    def chunks():
        yield b"a" * 12
        yield b"b" * 12

    with TestClient(app) as client:
        health_with_body = client.request("GET", "/health", content=b"x")
        invalid_chunked = client.post(
            "/embed",
            content=chunks(),
            headers={"content-type": "application/octet-stream"},
        )
    assert health_with_body.status_code == 413
    assert invalid_chunked.status_code == 413


def test_chunked_body_limit_is_enforced_without_content_length():
    received = [
        {"type": "http.request", "body": b"a" * 6, "more_body": True},
        {"type": "http.request", "body": b"b" * 6, "more_body": False},
    ]
    sent = []

    async def downstream(scope, receive, send):
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return received.pop(0)

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": [], "method": "POST", "path": "/embed"}
    middleware = RequestBodyLimitMiddleware(downstream, max_bytes=10)
    asyncio.run(middleware(scope, receive, send))
    assert sent[0]["status"] == 413


def test_upload_filename_is_never_used_as_a_filesystem_path(tmp_path):
    encoder = RecordingEncoder()
    app = create_app(encoder=encoder)
    target = tmp_path / "escaped.png"
    malicious_name = f"../../{target.name}"
    with TestClient(app) as client:
        response = client.post("/embed", files={"file": (malicious_name, png_bytes(), "image/png")})
    assert response.status_code == 200
    assert isinstance(encoder.received, bytes)
    assert not target.exists()


def test_optional_bearer_authentication():
    app = create_app(encoder=FakeEncoder(), api_key="a-long-test-api-key")
    files = {"file": ("signature.png", png_bytes(), "image/png")}
    with TestClient(app) as client:
        missing = client.post("/embed", files=files)
        wrong = client.post("/embed", files=files, headers={"Authorization": "Bearer wrong"})
        accepted = client.post(
            "/embed",
            files=files,
            headers={"Authorization": "Bearer a-long-test-api-key"},
        )
        unauthenticated_large = client.post("/embed", content=b"x" * (11 * 1024 * 1024))
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert accepted.status_code == 200
    # Authentication is evaluated before body limits or multipart parsing.
    assert unauthenticated_large.status_code == 401


def test_internal_exception_text_is_not_returned():
    app = create_app(encoder=LeakyEncoder())
    with TestClient(app) as client:
        response = client.post(
            "/embed", files={"file": ("signature.png", png_bytes(), "image/png")}
        )
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid or unsupported image"}
    assert "checkpoint" not in response.text


def test_request_concurrency_is_bounded_before_downstream_body_work():
    async def scenario():
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        first_sent = []
        second_sent = []

        async def downstream(scope, receive, send):
            first_started.set()
            await release_first.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def first_send(message):
            first_sent.append(message)

        async def second_send(message):
            second_sent.append(message)

        scope = {"type": "http", "headers": [], "method": "POST", "path": "/embed"}
        middleware = RequestConcurrencyMiddleware(downstream, capacity=1, timeout_seconds=0.01)
        first = asyncio.create_task(middleware(scope, receive, first_send))
        await first_started.wait()
        await middleware(scope, receive, second_send)
        release_first.set()
        await first
        return first_sent, second_sent

    first_sent, second_sent = asyncio.run(scenario())
    assert first_sent[0]["status"] == 200
    assert second_sent[0]["status"] == 503


def test_api_requires_checkpoint_digest_and_rejects_weak_keys(monkeypatch):
    monkeypatch.delenv("SIGORBIT_CHECKPOINT_SHA256", raising=False)
    app = create_app(checkpoint="/not/read/without/a/digest")
    with pytest.raises(RuntimeError, match="CHECKPOINT_SHA256"):
        with TestClient(app):
            pass
    with pytest.raises(ValueError, match="at least 16"):
        create_app(encoder=FakeEncoder(), api_key="too-short")
    with pytest.raises(ValueError, match="ASCII"):
        create_app(encoder=FakeEncoder(), api_key="é" * 16)


def test_console_refuses_unauthenticated_non_loopback_binding(monkeypatch):
    monkeypatch.setenv("SIGORBIT_HOST", "0.0.0.0")
    monkeypatch.delenv("SIGORBIT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="required for non-loopback"):
        api_module.run()
