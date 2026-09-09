"""Scaled orthogonal alignment used across condition-specific spaces."""

from __future__ import annotations

import torch
from scipy.linalg import orthogonal_procrustes as scipy_orthogonal_procrustes


def stable_seed(*items) -> int:
    text = "|".join(str(item) for item in items)
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % (2**31)


def random_orthogonal(k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(k, k, generator=generator), mode="reduced")
    return q.float()


def orthogonal_procrustes(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
    q, _ = scipy_orthogonal_procrustes(
        source.detach().cpu().numpy(),
        destination.detach().cpu().numpy(),
    )
    return torch.from_numpy(q).to(dtype=source.dtype)


def scaled_alpha(source: torch.Tensor, destination: torch.Tensor, q: torch.Tensor) -> float:
    aligned = source @ q
    denominator = aligned.square().sum().clamp_min(1e-12)
    return float((aligned * destination).sum() / denominator)


def coordinate_metrics(
    source: torch.Tensor,
    destination: torch.Tensor,
    q: torch.Tensor,
    alpha: float = 1.0,
) -> dict:
    if source.numel() == 0 or destination.numel() == 0:
        return {
            "n_alignment_samples": 0,
            "mean_squared_error": None,
            "root_mean_squared_error": None,
            "mean_cosine": None,
            "mean_norm_ratio": None,
            "median_norm_ratio": None,
        }
    aligned = float(alpha) * (source @ q)
    residual = aligned - destination
    cosine = torch.nn.functional.cosine_similarity(aligned, destination, dim=1)
    source_norm = source.norm(dim=1)
    destination_norm = destination.norm(dim=1)
    finite = source_norm > 1e-12
    ratio = destination_norm[finite] / source_norm[finite]
    return {
        "n_alignment_samples": int(source.shape[0]),
        "mean_squared_error": float(residual.square().mean()),
        "root_mean_squared_error": float(residual.square().mean().sqrt()),
        "mean_cosine": float(cosine.mean()),
        "mean_norm_ratio": float(ratio.mean()) if ratio.numel() else None,
        "median_norm_ratio": float(ratio.median()) if ratio.numel() else None,
    }
