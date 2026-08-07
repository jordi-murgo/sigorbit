"""Small FastAPI example exposing SigOrbit embeddings, without identity storage."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .encoder import SignatureEncoder


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
    device: str | None = None,
    encoder: SignatureEncoder | Any | None = None,
) -> FastAPI:
    """Application factory, injectable for tests and custom deployments."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if encoder is not None:
            app.state.encoder = encoder
        else:
            selected_checkpoint = checkpoint or os.getenv("SIGORBIT_CHECKPOINT")
            selected_device = device or os.getenv("SIGORBIT_DEVICE", "auto")
            app.state.encoder = SignatureEncoder(selected_checkpoint, device=selected_device)
        yield

    application = FastAPI(
        title="SigOrbit Embeddings",
        version="0.1.0",
        description=(
            "Example API for rotation-robust handwritten-signature embeddings. "
            "It accepts cropped signatures and does not perform identity decisions."
        ),
        lifespan=lifespan,
        license_info={"name": "MIT", "identifier": "MIT"},
    )

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
        data = await file.read()
        max_bytes = int(os.getenv("SIGORBIT_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
        if not data:
            raise HTTPException(400, "empty upload")
        if len(data) > max_bytes:
            raise HTTPException(413, f"image exceeds {max_bytes} bytes")
        active = application.state.encoder
        try:
            result = await run_in_threadpool(active.embed_with_details, data)
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"cannot decode or preprocess image: {exc}") from exc
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

    uvicorn.run(
        "sigorbit.api:app",
        host=os.getenv("SIGORBIT_HOST", "127.0.0.1"),
        port=int(os.getenv("SIGORBIT_PORT", "8000")),
        workers=1,
    )
