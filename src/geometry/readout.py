"""Utilities for separating output-facing and latent subspace directions."""

from __future__ import annotations

import torch

from src.geometry.subspaces import orthonormal_columns


def get_output_weight(model) -> torch.Tensor:
    output = model.get_output_embeddings()
    if output is not None and hasattr(output, "weight"):
        return output.weight.detach().float().cpu()
    for name in ("lm_head", "embed_out"):
        module = getattr(model, name, None)
        if module is not None and hasattr(module, "weight"):
            return module.weight.detach().float().cpu()
    raise ValueError("Could not find output embedding / lm_head weight on this model.")


def order_subspace_by_reference(
    basis: torch.Tensor, reference: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Rotate a basis from most to least aligned with a reference span."""
    d_model = max(basis.shape)
    source = orthonormal_columns(basis, d_model, "source basis")
    if source.shape[1] != min(basis.shape):
        raise ValueError("source basis is rank deficient")

    target = torch.as_tensor(reference).detach().float().squeeze()
    if target.ndim != 2:
        raise ValueError(f"reference basis must be 2D; got shape {tuple(target.shape)}")
    if target.shape[0] != d_model and target.shape[1] == d_model:
        target = target.T
    if target.shape[0] != d_model:
        raise ValueError(
            f"reference basis must have ambient dimension {d_model}; "
            f"got shape {tuple(target.shape)}"
        )
    _, singular_values, vh = torch.linalg.svd(target.T @ source, full_matrices=True)
    ordered = source @ vh.T
    if singular_values.numel() < ordered.shape[1]:
        singular_values = torch.cat(
            [
                singular_values,
                torch.zeros(
                    ordered.shape[1] - singular_values.numel(),
                    dtype=singular_values.dtype,
                    device=singular_values.device,
                ),
            ]
        )
    projector_error = float((ordered @ ordered.T - source @ source.T).norm())
    return ordered, singular_values.clamp(0, 1), projector_error


def split_subspace_by_reference(
    basis: torch.Tensor, reference: torch.Tensor, coupled_rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the readout-coupled and orthogonal coordinates of one subspace."""
    ordered, singular_values, _ = order_subspace_by_reference(basis, reference)
    if coupled_rank < 0 or coupled_rank > ordered.shape[1]:
        raise ValueError(f"coupled_rank must be in 0..{ordered.shape[1]}; got {coupled_rank}.")
    return ordered[:, :coupled_rank], ordered[:, coupled_rank:], singular_values
