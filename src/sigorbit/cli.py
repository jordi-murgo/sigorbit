"""Dependency-light console entry point for the optional FastAPI server."""

from importlib.util import find_spec


def run() -> None:
    missing = [name for name in ("fastapi", "multipart", "uvicorn") if find_spec(name) is None]
    if missing:
        raise SystemExit(
            "sigorbit-api requires optional dependencies; install 'sigorbit[api]' "
            f"(missing: {', '.join(missing)})"
        )
    from .api import run as run_api

    run_api()
