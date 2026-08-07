"""Neural network architecture used by SigOrbit.

The model learns a continuous SO(2) pose, resamples the input into that
canonical pose, and then applies a C8-steerable convolutional encoder. The
canonicalizer covers arbitrary angles while the C8 backbone supplies a strong
rotation-aware inductive bias. Reflections are intentionally not included.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from e2cnn import gspaces
from e2cnn import nn as enn
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """Serializable architecture configuration."""

    input_size: int = 257
    rotations: int = 8
    widths: tuple[int, int, int, int] = (24, 48, 96, 128)
    embedding_dim: int = 256
    dropout: float = 0.3

    def __post_init__(self) -> None:
        if self.input_size < 17 or self.input_size > 2049 or self.input_size % 2 == 0:
            raise ValueError("input_size must be an odd integer in [17, 2049]")
        if self.rotations < 2 or self.rotations > 32:
            raise ValueError("rotations must be in [2, 32]")
        if len(self.widths) != 4 or any(width < 1 or width > 1024 for width in self.widths):
            raise ValueError("widths must contain four integers in [1, 1024]")
        if self.embedding_dim < 1 or self.embedding_dim > 4096:
            raise ValueError("embedding_dim must be in [1, 4096]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_checkpoint_dict(self) -> dict[str, object]:
        data = asdict(self)
        return {
            "canonicalized": True,
            "in_size": data["input_size"],
            "N": data["rotations"],
            "widths": list(data["widths"]),
            "embedding_dim": data["embedding_dim"],
            "dropout": data["dropout"],
        }

    @classmethod
    def from_checkpoint_dict(cls, data: dict[str, object]) -> ModelConfig:
        if not data.get("canonicalized", False):
            raise ValueError("checkpoint is not a canonicalized SigOrbit model")
        widths = tuple(int(x) for x in data.get("widths", (24, 48, 96, 128)))
        if len(widths) != 4:
            raise ValueError("checkpoint widths must contain four stages")
        return cls(
            input_size=int(data.get("in_size", 257)),
            rotations=int(data.get("N", 8)),
            widths=widths,  # type: ignore[arg-type]
            embedding_dim=int(data.get("embedding_dim", 256)),
            dropout=float(data.get("dropout", 0.3)),
        )


class SteerableEncoder(nn.Module):
    """C_N-steerable CNN producing an L2-normalized invariant embedding."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gs = gspaces.Rot2dOnR2(N=config.rotations)
        self.in_type = enn.FieldType(self.gs, [self.gs.trivial_repr])

        def regular(fields: int) -> enn.FieldType:
            return enn.FieldType(self.gs, fields * [self.gs.regular_repr])

        t0, t1, t2, t3 = (regular(width) for width in config.widths)
        self.stem = enn.SequentialModule(
            enn.R2Conv(self.in_type, t0, kernel_size=7, padding=3, bias=False),
            enn.InnerBatchNorm(t0),
            enn.ReLU(t0, inplace=True),
            enn.PointwiseAvgPoolAntialiased(t0, sigma=0.66, stride=2),
        )
        self.layer1 = self._stage(t0, t1)
        self.layer2 = self._stage(t1, t2)
        self.layer3 = self._stage(t2, t3)
        self.gpool = enn.GroupPooling(t3)
        invariant_channels = self.gpool.out_type.size
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(invariant_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
        )

    @staticmethod
    def _stage(input_type: enn.FieldType, output_type: enn.FieldType) -> enn.SequentialModule:
        return enn.SequentialModule(
            enn.R2Conv(input_type, output_type, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(output_type),
            enn.ReLU(output_type, inplace=True),
            enn.R2Conv(output_type, output_type, kernel_size=5, padding=2, bias=False),
            enn.InnerBatchNorm(output_type),
            enn.ReLU(output_type, inplace=True),
            enn.PointwiseAvgPoolAntialiased(output_type, sigma=0.66, stride=2),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = enn.GeometricTensor(tensor, self.in_type)
        features = self.stem(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        invariant = self.gpool(features).tensor
        invariant = F.adaptive_avg_pool2d(invariant, 1)
        return F.normalize(self.head(invariant), p=2, dim=1)


class OrientationCanonicalizer(nn.Module):
    """Predict and remove a continuous planar rotation (no scale or reflection)."""

    def __init__(self):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, 2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 2),
        )
        nn.init.zeros_(self.loc[-1].weight)
        with torch.no_grad():
            self.loc[-1].bias.copy_(torch.tensor([1.0, 0.0]))

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.loc(tensor)
        cos_sin = cos_sin / (cos_sin.norm(dim=1, keepdim=True) + 1e-6)
        cosine, sine = cos_sin[:, 0], cos_sin[:, 1]
        rotation = torch.stack(
            (
                torch.stack((cosine, sine), dim=-1),
                torch.stack((-sine, cosine), dim=-1),
            ),
            dim=-2,
        )
        affine = torch.zeros(tensor.shape[0], 2, 3, device=tensor.device, dtype=tensor.dtype)
        affine[:, :, :2] = rotation
        grid = F.affine_grid(affine, tensor.shape, align_corners=False)
        canonical = F.grid_sample(
            tensor,
            grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=False,
        )
        return canonical, cos_sin


class CanonicalizedEncoder(nn.Module):
    """Continuous SO(2) canonicalizer followed by a C8 steerable encoder."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.canon = OrientationCanonicalizer()
        self.backbone = SteerableEncoder(self.config)

    def forward(
        self, tensor: torch.Tensor, *, return_orientation: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        canonical, cos_sin = self.canon(tensor)
        embedding = self.backbone(canonical)
        if return_orientation:
            return embedding, cos_sin
        return embedding
