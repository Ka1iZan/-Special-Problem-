from __future__ import annotations
import torch
import torch.nn.functional as F

ENTITY_TYPES = ("PER", "LOC", "ORG")


def mean_pool_span(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Mean-pool selected token positions from [seq, hidden]."""
    if positions.numel() == 0:
        raise ValueError("Cannot pool an empty entity span")
    return hidden.index_select(0, positions).mean(dim=0)


def centroid(vectors: torch.Tensor) -> torch.Tensor:
    """Return class centroid for [n, hidden]."""
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Expected non-empty [n, hidden] tensor")
    return vectors.float().mean(dim=0)


def shared_prototype(language_prototypes: torch.Tensor) -> torch.Tensor:
    """Equal-language-weight mean over [n_languages, hidden]."""
    return centroid(language_prototypes)


def cosine_drift(current: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(current.float(), anchor.float(), dim=0)


def prototype_logits(entity_vectors: torch.Tensor, prototypes: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Cosine-similarity logits: [batch, hidden] x [classes, hidden]."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    z = F.normalize(entity_vectors.float(), dim=-1)
    p = F.normalize(prototypes.float(), dim=-1)
    return (z @ p.T) / tau


def alignment_loss(entity_vectors: torch.Tensor, prototypes: torch.Tensor, class_ids: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    return F.cross_entropy(prototype_logits(entity_vectors, prototypes, tau), class_ids)


def drift_sampling_probs(drifts: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return torch.softmax(drifts.float() / tau, dim=0)
