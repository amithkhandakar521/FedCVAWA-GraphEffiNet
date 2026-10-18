
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from .client import ClientUpdate, flatten_parameters, load_flat_parameters


@dataclass
class FedCVAWAConfig:
    """Aggregation hyperparameters. Defaults match paper Table 1."""

    tau_conv: float = 0.7
    alpha: float = 0.5
    tau_max: float = 1.0
    tau_min: float = 0.1
    total_rounds: int = 100


@dataclass
class ClientReport:
    client_id: int
    convergence_score: float
    accepted: bool
    stability: float
    val_accuracy: float | None = None
    quality_score: float | None = None
    weight: float = 0.0


@dataclass
class RoundReport:
    round_idx: int
    temperature: float
    global_norm: float
    clients: list[ClientReport] = field(default_factory=list)

    @property
    def num_accepted(self) -> int:
        return sum(c.accepted for c in self.clients)


def convergence_score(update_norm: float, global_norm: float) -> float:
    """Eq. 9. Returns a value in [0, 1]; 1 means no displacement."""
    if global_norm <= 0.0:
        raise ValueError("Global parameter norm must be positive.")
    return 1.0 - min(update_norm / global_norm, 1.0)


def cosine_temperature(round_idx: int, cfg: FedCVAWAConfig) -> float:
    """Eq. 14: tau_max at t = 0, tau_min at t = T."""
    progress = math.cos(math.pi * round_idx / cfg.total_rounds)
    return cfg.tau_min + 0.5 * (cfg.tau_max - cfg.tau_min) * (1.0 + progress)


def quality_score(stability: float, val_accuracy: float, alpha: float) -> float:
    """Eq. 12."""
    return alpha * stability + (1.0 - alpha) * val_accuracy


def softmax_weights(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """Eq. 13."""
    return torch.softmax(scores / temperature, dim=0)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    """Top-1 accuracy on ``loader`` (Eq. 11)."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


class FedCVAWA:
    """Server-side FedCVAWA aggregator."""

    def __init__(self, config: FedCVAWAConfig | None = None) -> None:
        self.config = config or FedCVAWAConfig()
        if not 0.0 <= self.config.tau_conv <= 1.0:
            raise ValueError("tau_conv must lie in [0, 1].")
        if not 0.0 <= self.config.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        if not 0.0 < self.config.tau_min <= self.config.tau_max:
            raise ValueError("Require 0 < tau_min <= tau_max.")

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
        round_idx: int,
        val_loader: DataLoader,
        eval_model: nn.Module,
        device: torch.device,
    ) -> RoundReport:
        """Update ``global_model`` in place and return a per-client report.

        ``eval_model`` is a scratch model with the same architecture, used to
        evaluate each candidate ``w + delta_k`` on the server validation set.
        """
        cfg = self.config
        global_vector = flatten_parameters(global_model).to(device)
        global_norm = float(torch.linalg.vector_norm(global_vector.float()))
        report = RoundReport(
            round_idx=round_idx,
            temperature=cosine_temperature(round_idx, cfg),
            global_norm=global_norm,
        )

        # Stage 1: convergence verification.
        accepted: list[ClientUpdate] = []
        for update in updates:
            score = convergence_score(update.sparse_delta.norm(), global_norm)
            is_accepted = score >= cfg.tau_conv
            report.clients.append(
                ClientReport(
                    client_id=update.client_id,
                    convergence_score=score,
                    accepted=is_accepted,
                    stability=update.stability,
                )
            )
            if is_accepted:
                accepted.append(update)
        if not accepted:
            return report  # w^{t+1} = w^{t}

        # Stage 2: server-side validation of each candidate model.
        by_id = {c.client_id: c for c in report.clients}
        eval_model.load_state_dict(global_model.state_dict())
        eval_model.to(device)
        scores = []
        for update in accepted:
            dense_delta = update.sparse_delta.to_dense(device)
            load_flat_parameters(eval_model, global_vector + dense_delta)
            _load_buffers(eval_model, update.buffers, device)
            accuracy = evaluate_accuracy(eval_model, val_loader, device)
            entry = by_id[update.client_id]
            entry.val_accuracy = accuracy
            entry.quality_score = quality_score(update.stability, accuracy, cfg.alpha)
            scores.append(entry.quality_score)

        weights = softmax_weights(
            torch.tensor(scores, dtype=torch.float64), report.temperature
        )
        aggregated = torch.zeros_like(global_vector)
        for weight, update in zip(weights.tolist(), accepted):
            by_id[update.client_id].weight = weight
            aggregated += weight * update.sparse_delta.to_dense(device)
        load_flat_parameters(global_model, global_vector + aggregated)
        _aggregate_buffers(global_model, accepted, weights.tolist(), device)
        return report


@torch.no_grad()
def _load_buffers(
    model: nn.Module, buffers: dict[str, torch.Tensor], device: torch.device
) -> None:
    named = dict(model.named_buffers())
    for name, value in buffers.items():
        named[name].copy_(value.to(device))


@torch.no_grad()
def _aggregate_buffers(
    model: nn.Module,
    updates: list[ClientUpdate],
    weights: list[float],
    device: torch.device,
) -> None:
    """Weighted average of BatchNorm running statistics.

    The paper aggregates trainable parameters only. BatchNorm running
    statistics are not trainable, so they are averaged with the same
    FedCVAWA weights to keep them consistent with the aggregated parameters.
    """
    named = dict(model.named_buffers())
    for name in updates[0].buffers:
        stacked = sum(w * u.buffers[name].to(device) for w, u in zip(weights, updates))
        named[name].copy_(stacked)
