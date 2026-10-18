"""FedCVAWA: Federated Convergence-Verified Adaptive Weighted Aggregation."""

from .aggregation import (
    ClientReport,
    FedCVAWA,
    FedCVAWAConfig,
    RoundReport,
    convergence_score,
    cosine_temperature,
    evaluate_accuracy,
    quality_score,
    softmax_weights,
)
from .client import (
    ClientUpdate,
    LocalTrainingConfig,
    SparseUpdate,
    flatten_parameters,
    load_flat_parameters,
    local_update,
    topk_sparsify,
)

__version__ = "1.0.0"

__all__ = [
    "ClientReport",
    "ClientUpdate",
    "FedCVAWA",
    "FedCVAWAConfig",
    "LocalTrainingConfig",
    "RoundReport",
    "SparseUpdate",
    "convergence_score",
    "cosine_temperature",
    "evaluate_accuracy",
    "flatten_parameters",
    "load_flat_parameters",
    "local_update",
    "quality_score",
    "softmax_weights",
    "topk_sparsify",
]
