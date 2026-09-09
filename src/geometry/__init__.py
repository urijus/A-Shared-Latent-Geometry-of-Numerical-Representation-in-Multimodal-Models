"""Geometry utilities shared by the numerical-representation experiments."""

from src.geometry.alignment import (
    coordinate_metrics,
    orthogonal_procrustes,
    random_orthogonal,
    scaled_alpha,
    stable_seed,
)
from src.geometry.synchronization import (
    hub_map,
    project_orthogonal,
    rotation_residuals,
    synchronize_rotations,
    synchronize_scales,
)
from src.geometry.readout import (
    get_output_weight,
    order_subspace_by_reference,
    split_subspace_by_reference,
)

__all__ = [
    "coordinate_metrics",
    "get_output_weight",
    "hub_map",
    "orthogonal_procrustes",
    "order_subspace_by_reference",
    "project_orthogonal",
    "random_orthogonal",
    "rotation_residuals",
    "scaled_alpha",
    "stable_seed",
    "split_subspace_by_reference",
    "synchronize_rotations",
    "synchronize_scales",
]
