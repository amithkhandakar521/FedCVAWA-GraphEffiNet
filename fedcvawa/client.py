
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Trainable parameters in a fixed order shared by clients and server."""
    return [p for p in model.parameters() if p.requires_grad]


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    """Return the trainable parameters of ``model`` as one detached vector."""
    return torch.cat([p.detach().reshape(-1) for p in trainable_parameters(model)])


@torch.no_grad()
def load_flat_parameters(model: nn.Module, vector: torch.Tensor) -> None:
    """Copy a flat vector back into the trainable parameters of ``model``."""
    offset = 0
    for param in trainable_parameters(model):
        numel = param.numel()
        param.copy_(vector[offset:offset + numel].view_as(param))
        offset += numel
    if offset != vector.numel():
        raise ValueError(
            f"Vector has {vector.numel()} entries, model expects {offset}."
        )


def float_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    """Floating point buffers (BatchNorm running statistics)."""
    return {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
        if buf.is_floating_point() and name in model.state_dict()
    }


@dataclass
class SparseUpdate:
    """Value-index encoding of a top-K sparsified update."""

    values: torch.Tensor
    indices: torch.Tensor
    numel: int

    @property
    def k(self) -> int:
        return self.values.numel()

    @property
    def compression_ratio(self) -> float:
        """r_comp = K / |theta| (Eq. 7)."""
        return self.k / self.numel

    def norm(self) -> float:
        """L2 norm of the sparsified update (Eq. 8)."""
        return float(torch.linalg.vector_norm(self.values.float()))

    def to_dense(self, device: torch.device | str | None = None) -> torch.Tensor:
        device = device or self.values.device
        dense = torch.zeros(self.numel, dtype=self.values.dtype, device=device)
        dense[self.indices.to(device)] = self.values.to(device)
        return dense

    def payload_bytes(self, include_indices: bool = True) -> int:
        """Bytes on the wire: float32 values plus int32 indices."""
        index_bytes = 4 if self.numel < 2**31 else 8
        return self.k * (4 + (index_bytes if include_indices else 0))


def topk_sparsify(delta: torch.Tensor, ratio: float) -> SparseUpdate:
    """Keep the ``ratio * |theta|`` largest-magnitude entries of ``delta``."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"Sparsification ratio must be in (0, 1], got {ratio}.")
    flat = delta.reshape(-1)
    k = max(1, int(round(ratio * flat.numel())))
    _, indices = torch.topk(flat.abs(), k, sorted=False)
    return SparseUpdate(
        values=flat[indices].clone(), indices=indices, numel=flat.numel()
    )


@dataclass
class LocalTrainingConfig:
    """Local optimization settings (paper Section 4.6)."""

    local_epochs: int = 100
    lr_max: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    sparsity_ratio: float = 0.20
    amp: bool = True
    label_smoothing: float = 0.0


@dataclass
class ClientUpdate:
    """What a client sends to the server in one round."""

    client_id: int
    sparse_delta: SparseUpdate
    stability: float  # g_tilde = 1 / (1 + ||grad L_k||) (Eq. 10)
    grad_norm: float
    buffers: dict[str, torch.Tensor]
    num_samples: int
    train_loss: float

    def payload_bytes(self) -> int:
        buffer_bytes = sum(b.numel() * 4 for b in self.buffers.values())
        return self.sparse_delta.payload_bytes() + buffer_bytes + 4


def _total_grad_norm(model: nn.Module) -> float:
    grads = [p.grad.detach().reshape(-1) for p in model.parameters()
             if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(grads).float()))


def local_update(
    client_id: int,
    model: nn.Module,
    global_state: dict[str, torch.Tensor],
    loader: DataLoader,
    config: LocalTrainingConfig,
    device: torch.device,
) -> ClientUpdate:
    """Client side of one FedCVAWA round (Algorithm 2, lines 5 to 10).

    Starts from the global model, runs ``E`` local epochs with AdamW and a
    per-epoch cosine learning rate (Eq. 30), records the gradient norm at the
    final local step (Eq. 10), and returns the top-K sparsified update.

    ``model`` is a reusable working copy. It is reset to ``global_state``
    first, so one instance can serve every client.
    """
    model.load_state_dict(global_state)
    model.to(device).train()
    global_vector = flatten_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr_max, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    grad_norm, running_loss, seen = 0.0, 0.0, 0
    for epoch in range(config.local_epochs):
        lr = config.lr_min + 0.5 * (config.lr_max - config.lr_min) * (
            1 + math.cos(math.pi * epoch / config.local_epochs)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        running_loss, seen = 0.0, 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = _total_grad_norm(model)  # kept for the final step
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

    delta = flatten_parameters(model) - global_vector
    sparse = topk_sparsify(delta, config.sparsity_ratio)
    return ClientUpdate(
        client_id=client_id,
        sparse_delta=SparseUpdate(sparse.values.cpu(), sparse.indices.cpu(),
                                  sparse.numel),
        stability=1.0 / (1.0 + grad_norm),
        grad_norm=grad_norm,
        buffers={k: v.cpu() for k, v in float_buffers(model).items()},
        num_samples=seen,
        train_loss=running_loss / max(seen, 1),
    )
