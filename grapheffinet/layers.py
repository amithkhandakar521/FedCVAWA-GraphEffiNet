

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_normalized_adjacency(grid_size: int) -> torch.Tensor:
    """Return D^{-1/2} (A + I) D^{-1/2} for a grid graph with 8-connectivity.

    Nodes are indexed in row-major order, matching the patch order produced
    by :func:`patchify`. Two nodes are adjacent when the Chebyshev distance
    between their grid positions is 1 (Eq. 17).
    """
    num_nodes = grid_size * grid_size
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
        ),
        dim=-1,
    ).reshape(num_nodes, 2)
    chebyshev = (coords[:, None, :] - coords[None, :, :]).abs().amax(dim=-1)
    adjacency = (chebyshev <= 1).float()  # includes self-loops (A + I)
    degree_inv_sqrt = adjacency.sum(dim=1).pow(-0.5)
    return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split ``(B, C, H, W)`` images into ``(B * N, C, P, P)`` patches (Eq. 16)."""
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Image size {height}x{width} is not divisible by patch size "
            f"{patch_size}."
        )
    patches = images.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )  # (B, C, gh, gw, P, P)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(-1, channels, patch_size, patch_size)


class PatchEncoder(nn.Module):
    """Three-layer convolutional encoder (3 -> 64 -> 128 -> out_dim) + GAP."""

    def __init__(self, in_channels: int = 3, out_dim: int = 256) -> None:
        super().__init__()
        widths = (in_channels, 64, 128, out_dim)
        layers: list[nn.Module] = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            layers += [
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(patches)).flatten(1)


class GCNLayer(nn.Module):
    """H' = ReLU(BN(A_hat H W + b)) (Eq. 19)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, _ = nodes.shape
        out = torch.matmul(adjacency, self.linear(nodes)) + self.bias
        out = self.norm(out.reshape(batch * num_nodes, -1))
        return F.relu(out).view(batch, num_nodes, -1)


def _window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window, window, width // window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)


def _window_reverse(
    windows: torch.Tensor, window: int, height: int, width: int
) -> torch.Tensor:
    num_windows = (height // window) * (width // window)
    batch = windows.shape[0] // num_windows
    x = windows.view(batch, height // window, width // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention within a window, with relative position bias."""

    def __init__(self, dim: int, window: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} is not divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")
        ).flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
        relative = relative + (window - 1)
        index = relative[..., 0] * (2 * window - 1) + relative[..., 1]
        self.register_buffer("relative_index", index, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        batch_windows, tokens, dim = x.shape
        qkv = self.qkv(x).view(batch_windows, tokens, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_bias_table[self.relative_index.view(-1)]
        attn = attn + bias.view(tokens, tokens, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(-1, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask[None, :, None]
            attn = attn.view(-1, self.num_heads, tokens, tokens)
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(
            batch_windows, tokens, dim
        )
        return self.proj(out)


class SwinBlock(nn.Module):
    """Swin Transformer block over the patch grid (Eqs. 20 to 22)."""

    def __init__(
        self,
        dim: int,
        grid_size: int,
        num_heads: int,
        window: int,
        shift: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if grid_size % window:
            raise ValueError(
                f"Patch grid {grid_size} is not divisible by window {window}."
            )
        if window >= grid_size:
            window, shift = grid_size, 0
        self.grid_size, self.window, self.shift = grid_size, window, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.register_buffer("attn_mask", self._build_mask(), persistent=False)

    def _build_mask(self) -> torch.Tensor | None:
        if self.shift == 0:
            return None
        size, window, shift = self.grid_size, self.window, self.shift
        region = torch.zeros(1, size, size, 1)
        slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
        label = 0
        for rows in slices:
            for cols in slices:
                region[:, rows, cols, :] = label
                label += 1
        windows = _window_partition(region, window).squeeze(-1)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, dim = x.shape
        size = self.grid_size
        h = self.norm1(x).view(batch, size, size, dim)
        if self.shift:
            h = torch.roll(h, shifts=(-self.shift, -self.shift), dims=(1, 2))
        h = self.attn(_window_partition(h, self.window), self.attn_mask)
        h = _window_reverse(h, self.window, size, size)
        if self.shift:
            h = torch.roll(h, shifts=(self.shift, self.shift), dims=(1, 2))
        x = x + h.reshape(batch, num_nodes, dim)
        return x + self.mlp(self.norm2(x))


class GraphBranch(nn.Module):
    """Patch graph -> GCN -> Swin blocks -> pooled projection (Eq. 23)."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 56,
        gcn_dim: int = 256,
        gcn_layers: int = 3,
        swin_blocks: int = 2,
        num_heads: int = 8,
        window_size: int = 2,
        fusion_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size.")
        self.patch_size = patch_size
        grid_size = image_size // patch_size
        self.register_buffer(
            "adjacency", build_normalized_adjacency(grid_size), persistent=False
        )
        self.encoder = PatchEncoder(out_dim=gcn_dim)
        self.gcn = nn.ModuleList(
            GCNLayer(gcn_dim, gcn_dim) for _ in range(gcn_layers)
        )
        self.swin = nn.ModuleList(
            SwinBlock(
                gcn_dim,
                grid_size,
                num_heads,
                window_size,
                shift=0 if i % 2 == 0 else window_size // 2,
                dropout=dropout,
            )
            for i in range(swin_blocks)
        )
        self.project = nn.Sequential(
            nn.Linear(gcn_dim, fusion_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        nodes = self.encoder(patchify(images, self.patch_size))
        nodes = nodes.view(batch, -1, nodes.shape[-1])
        for layer in self.gcn:
            nodes = layer(nodes, self.adjacency)
        for block in self.swin:
            nodes = block(nodes)
        return self.project(nodes.mean(dim=1))


class SpatialSelfAttention(nn.Module):
    """Residual spatial self-attention over a feature map (Eqs. 24 and 25).

    Queries and keys are projected to ``channels // reduction`` dimensions with
    1x1 convolutions, values keep ``channels`` dimensions. The output is added
    to the input through a learnable scalar ``gamma`` initialized to zero, so
    the residual path dominates at the start of training.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        qk_dim = max(1, channels // reduction)
        self.query = nn.Conv2d(channels, qk_dim, kernel_size=1)
        self.key = nn.Conv2d(channels, qk_dim, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = qk_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q = self.query(x).flatten(2).transpose(1, 2)  # (B, HW, d)
        k = self.key(x).flatten(2)  # (B, d, HW)
        v = self.value(x).flatten(2).transpose(1, 2)  # (B, HW, C)
        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, channels, height, width)
        return self.gamma * out + x
