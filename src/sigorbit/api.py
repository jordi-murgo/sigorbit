"""Small FastAPI example exposing SigOrbit embeddings, without identity storage."""

from __future__ import annotations

import asyncio
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .encoder import SignatureEncoder
from .preprocessing import DEFAULT_MAX_IMAGE_PIXELS

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_MULTIPART_OVERHEAD_BYTES = 64 * 1024


async def _send_json_error(
    send: Send,
    status: int,
    detail: str,
    *,
    extra_headers: tuple[tuple[bytes, bytes], ...] = (),
) -> None:
    body = ('{"detail":"' + detail + '"}').encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        *extra_headers,
    ]
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestBodyLimitMiddleware:
    """Buffer and bound raw bodies before routing or multipart parsing.

    Only ``/embed`` accepts a body. Other routes reject any non-empty body.
    The outer full-request semaphore bounds aggregate buffering for embeddings.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body_limit = self.max_bytes if scope.get("path") == "/embed" else 0
        content_lengths = [value for key, value in scope["headers"] if key == b"content-length"]
        if content_lengths:
            try:
                content_length = int(content_lengths[-1])
            except ValueError:
                await _send_json_error(
                    send, 400, "invalid Content-Length", extra_headers=((b"connection", b"close"),)
                )
                return
            if content_length < 0:
                await _send_json_error(
                    send, 400, "invalid Content-Length", extra_headers=((b"connection", b"close"),)
                )
                return
            if content_length > body_limit:
                await _send_json_error(
                    send, 413, "request body too large", extra_headers=((b"connection", b"close"),)
                )
                return

        messages: list[Message] = []
        received = 0
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                break
            if message["type"] != "http.request":
                continue
            received += len(message.get("body", b""))
            if received > body_limit:
                await _send_json_error(
                    send, 413, "request body too large", extra_headers=((b"connection", b"close"),)
                )
                return
            if not message.get("more_body", False):
                break

        index = 0

        async def replay_receive() -> Message:
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)


class BearerAuthMiddleware:
    """Authenticate embedding requests before reading or parsing their bodies."""

    def __init__(self, app: ASGIApp, *, api_key: str):
        self.app = app
        self.expected = f"Bearer {api_key}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") == "/embed":
            values = [value for key, value in scope["headers"] if key == b"authorization"]
            supplied = values[0] if len(values) == 1 else b""
            if not secrets.compare_digest(supplied, self.expected):
                await _send_json_error(
                    send,
                    401,
                    "invalid or missing bearer token",
                    extra_headers=(
                        (b"www-authenticate", b"Bearer"),
                        (b"connection", b"close"),
                    ),
                )
                return
        await self.app(scope, receive, send)


class RequestConcurrencyMiddleware:
    """Bound full upload-to-response work, including multipart parsing."""

    def __init__(self, app: ASGIApp, *, capacity: int, timeout_seconds: float):
        self.app = app
        self.slots = asyncio.Semaphore(capacity)
        self.timeout_seconds = timeout_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/embed":
            await self.app(scope, receive, send)
            return
        try:
            await asyncio.wait_for(self.slots.acquire(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            await _send_json_error(
                send,
                503,
                "request capacity is busy; retry later",
                extra_headers=((b"connection", b"close"),),
            )
            return
        try:
            await self.app(scope, receive, send)
        finally:
            self.slots.release()


@dataclass(frozen=True)
class ApiLimits:
    max_upload_bytes: int
    max_request_bytes: int
    max_image_pixels: int
    max_concurrent_requests: int
    queue_timeout_seconds: float


def _positive_int(value: int | str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_float(value: float | str, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _resolve_limits(
    *,
    max_upload_bytes: int | None,
    max_request_bytes: int | None,
    max_image_pixels: int | None,
    max_concurrent_requests: int | None,
    queue_timeout_seconds: float | None,
) -> ApiLimits:
    upload = _positive_int(
        max_upload_bytes
        if max_upload_bytes is not None
        else os.getenv("SIGORBIT_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
        "max_upload_bytes",
    )
    request = _positive_int(
        max_request_bytes
        if max_request_bytes is not None
        else os.getenv("SIGORBIT_MAX_REQUEST_BYTES", upload + DEFAULT_MULTIPART_OVERHEAD_BYTES),
        "max_request_bytes",
    )
    if request <= upload:
        raise ValueError("max_request_bytes must exceed max_upload_bytes for multipart framing")
    pixels = _positive_int(
        max_image_pixels
        if max_image_pixels is not None
        else os.getenv("SIGORBIT_MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS),
        "max_image_pixels",
    )
    concurrent = _positive_int(
        max_concurrent_requests
        if max_concurrent_requests is not None
        else os.getenv("SIGORBIT_MAX_CONCURRENT_REQUESTS", 1),
        "max_concurrent_requests",
    )
    timeout = _positive_float(
        queue_timeout_seconds
        if queue_timeout_seconds is not None
        else os.getenv("SIGORBIT_QUEUE_TIMEOUT_SECONDS", 1.0),
        "queue_timeout_seconds",
    )
    return ApiLimits(upload, request, pixels, concurrent, timeout)


class HealthResponse(BaseModel):
    status: str
    model_id: str
    preprocess_version: str
    input_size: int
    embedding_dim: int
    device: str


class EmbeddingResponse(BaseModel):
    model_id: str
    preprocess_version: str
    input_size: int
    dimensions: int
    normalized: bool
    orientation_degrees: float
    embedding: list[float] = Field(description="L2-normalized float32 embedding")


def create_app(
    *,
    checkpoint: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    device: str | None = None,
    encoder: SignatureEncoder | Any | None = None,
    api_key: str | None = None,
    max_upload_bytes: int | None = None,
    max_request_bytes: int | None = None,
    max_image_pixels: int | None = None,
    max_concurrent_requests: int | None = None,
    queue_timeout_seconds: float | None = None,
) -> FastAPI:
    """Application factory, injectable for tests and custom deployments."""
    limits = _resolve_limits(
        max_upload_bytes=max_upload_bytes,
        max_request_bytes=max_request_bytes,
        max_image_pixels=max_image_pixels,
        max_concurrent_requests=max_concurrent_requests,
        queue_timeout_seconds=queue_timeout_seconds,
    )
    configured_api_key = api_key if api_key is not None else os.getenv("SIGORBIT_API_KEY")
    if configured_api_key is not None:
        if len(configured_api_key) < 16:
            raise ValueError("SIGORBIT_API_KEY must contain at least 16 characters")
        try:
            configured_api_key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("SIGORBIT_API_KEY must contain only ASCII characters") from exc

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if encoder is not None:
            app.state.encoder = encoder
        else:
            selected_checkpoint = checkpoint or os.getenv("SIGORBIT_CHECKPOINT")
            selected_device = device or os.getenv("SIGORBIT_DEVICE", "auto")
            selected_sha256 = expected_checkpoint_sha256 or os.getenv("SIGORBIT_CHECKPOINT_SHA256")
            if selected_sha256 is None:
                raise RuntimeError("SIGORBIT_CHECKPOINT_SHA256 is required by the HTTP API")
            app.state.encoder = SignatureEncoder(
                selected_checkpoint,
                device=selected_device,
                expected_checkpoint_sha256=selected_sha256,
                max_image_pixels=limits.max_image_pixels,
            )
        yield

    application = FastAPI(
        title="SigOrbit Embeddings",
        version="0.2.4",
        description=(
            "Example API for rotation-robust handwritten-signature embeddings. "
            "It accepts cropped signatures and does not perform identity decisions."
        ),
        lifespan=lifespan,
        license_info={"name": "MIT", "identifier": "MIT"},
    )
    application.add_middleware(RequestBodyLimitMiddleware, max_bytes=limits.max_request_bytes)
    application.add_middleware(
        RequestConcurrencyMiddleware,
        capacity=limits.max_concurrent_requests,
        timeout_seconds=limits.queue_timeout_seconds,
    )
    if configured_api_key is not None:
        application.add_middleware(BearerAuthMiddleware, api_key=configured_api_key)

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        active = application.state.encoder
        return HealthResponse(
            status="ok",
            model_id=active.model_id,
            preprocess_version=active.preprocess_version,
            input_size=active.input_size,
            embedding_dim=active.embedding_dim,
            device=str(active.device),
        )

    @application.post(
        "/embed",
        response_model=EmbeddingResponse,
        tags=["embeddings"],
        summary="Generate an embedding from one cropped signature",
    )
    async def embed(
        file: Annotated[UploadFile, File(description="PNG/JPEG/WebP signature crop")],
    ) -> EmbeddingResponse:
        try:
            data = await file.read(limits.max_upload_bytes + 1)
        finally:
            await file.close()
        if not data:
            raise HTTPException(400, "empty upload")
        if len(data) > limits.max_upload_bytes:
            raise HTTPException(413, "uploaded image too large")

        active = application.state.encoder
        try:
            result = await run_in_threadpool(active.embed_with_details, data)
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(400, "invalid or unsupported image") from exc

        vector = np.asarray(result.embedding, dtype=np.float32)
        if vector.shape != (active.embedding_dim,) or not np.isfinite(vector).all():
            raise HTTPException(500, "model returned an invalid embedding")
        return EmbeddingResponse(
            model_id=active.model_id,
            preprocess_version=active.preprocess_version,
            input_size=active.input_size,
            dimensions=active.embedding_dim,
            normalized=bool(np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4)),
            orientation_degrees=round(float(result.orientation_degrees), 4),
            embedding=vector.tolist(),
        )

    return application


app = create_app()


def run() -> None:
    """Console entry point installed by the ``api`` extra."""
    import uvicorn

    host = os.getenv("SIGORBIT_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "::1", "localhost"} and os.getenv("SIGORBIT_API_KEY") is None:
        raise RuntimeError("SIGORBIT_API_KEY is required for non-loopback binding")
    uvicorn.run(
        "sigorbit.api:app",
        host=host,
        port=int(os.getenv("SIGORBIT_PORT", "8000")),
        workers=1,
    )
