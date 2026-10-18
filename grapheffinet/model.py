

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .layers import GraphBranch, SpatialSelfAttention

# Output channels of torchvision ``efficientnet_b0().features[k]``.
# "MBConv-k" in the paper refers to the output of ``features[k]``.
STAGE_CHANNELS = {1: 16, 2: 24, 3: 40, 4: 80, 5: 112, 6: 192, 7: 320}
BACKBONE_OUT_CHANNELS = 1280
CLASSES = ("normal", "bump", "pothole")


@dataclass
class ModelConfig:
    """Architecture hyperparameters. Defaults match the paper (S-16)."""

    num_classes: int = 3
    image_size: int = 224
    patch_size: int = 56
    gcn_dim: int = 256
    gcn_layers: int = 3
    swin_blocks: int = 2
    num_heads: int = 8
    window_size: int = 2
    attention_position: int | None = 4
    fusion_dim: int = 512
    head_hidden: int = 512
    dropout: float = 0.3
    use_graph_branch: bool = True
    pretrained: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path, section: str = "model") -> "ModelConfig":
        """Load the ``model`` section of a YAML file (requires PyYAML)."""
        import yaml

        values = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**values.get(section, values))


class EfficientNetBranch(nn.Module):
    """EfficientNet-B0 features, optional spatial self-attention, projection."""

    def __init__(
        self,
        attention_position: int | None = 4,
        fusion_dim: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if attention_position is not None and attention_position not in STAGE_CHANNELS:
            raise ValueError(
                f"attention_position must be one of {sorted(STAGE_CHANNELS)} "
                "or None."
            )
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        features = efficientnet_b0(weights=weights).features

        if attention_position:
            split = attention_position + 1
            self.stem = features[:split]
            self.attention = SpatialSelfAttention(STAGE_CHANNELS[attention_position])
            self.tail = features[split:]
        else:
            self.stem, self.attention, self.tail = nn.Identity(), nn.Identity(), features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(
            nn.Linear(BACKBONE_OUT_CHANNELS, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.tail(self.attention(self.stem(images)))
        return self.project(self.pool(x).flatten(1))


class GraphEffiNet(nn.Module):
    """Concatenation fusion of the graph branch and the EfficientNet branch.

    ``forward`` expects ImageNet-normalized ``(B, 3, 224, 224)`` images and
    returns class logits. Setting ``use_graph_branch=False`` reproduces the
    S-0 ablation.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.graph_branch = (
            GraphBranch(
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                gcn_dim=cfg.gcn_dim,
                gcn_layers=cfg.gcn_layers,
                swin_blocks=cfg.swin_blocks,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                fusion_dim=cfg.fusion_dim,
                dropout=cfg.dropout,
            )
            if cfg.use_graph_branch
            else None
        )
        self.efficientnet_branch = EfficientNetBranch(
            attention_position=cfg.attention_position,
            fusion_dim=cfg.fusion_dim,
            dropout=cfg.dropout,
            pretrained=cfg.pretrained,
        )
        fused_dim = cfg.fusion_dim * (2 if cfg.use_graph_branch else 1)
        self.head = nn.Sequential(
            nn.Linear(fused_dim, cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = [self.efficientnet_branch(images)]
        if self.graph_branch is not None:
            features.insert(0, self.graph_branch(images))
        return self.head(torch.cat(features, dim=1))


def build_model(config: ModelConfig | dict | None = None) -> GraphEffiNet:
    """Build GraphEffiNet from a :class:`ModelConfig` or a plain dict."""
    if isinstance(config, dict):
        config = ModelConfig(**config)
    return GraphEffiNet(config)
