#!/usr/bin/env python3
"""Create a portable inference checkpoint without derived e2cnn filter caches."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

DERIVED = (".filter", ".expanded_bias")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=True)
    state = {
        key: value for key, value in source["model_state_dict"].items() if not key.endswith(DERIVED)
    }
    package = {
        "format_version": 1,
        "model_state_dict": state,
        "config": source["config"],
        "metadata": {
            "model_id": "sigorbit-c8-257-v1",
            "architecture": "so2-canonicalized-c8",
            "preprocess_version": "sigorbit-gray-square-257-v1",
            "weights_release_status": "withheld_pending_upstream_permission",
            "training_dataset": "rakshitdabral/Signature-Verification-Dataset",
            "training_dataset_card_declared_license": "MIT",
            "upstream_dataset_license_status": "CEDAR_unlicensed_BHSig260_research_only",
            "source_checkpoint_sha256": sha256(args.source),
            "selected_epoch_zero_based": int(source.get("epoch", -1)),
            "validation_top1": float(source.get("val_top1", float("nan"))),
            "validation_margin": float(source.get("val_margin", float("nan"))),
        },
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.destination)
    print(f"wrote {args.destination} ({args.destination.stat().st_size / 2**20:.1f} MiB)")
    print(f"sha256 {sha256(args.destination)}")


if __name__ == "__main__":
    main()
