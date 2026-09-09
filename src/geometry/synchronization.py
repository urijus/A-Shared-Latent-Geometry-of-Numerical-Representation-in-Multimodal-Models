"""Orthogonal and scale synchronization over pairwise geometry maps."""

from __future__ import annotations

import statistics
from typing import Protocol

import torch


class AlignmentEdge(Protocol):
    source: str
    destination: str
    q: torch.Tensor
    alpha: float
    weight: float
    path: object


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sample_std(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.stdev(values) if len(values) > 1 else 0.0


def project_orthogonal(matrix: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def rotation_residuals(
    tasks: list[str],
    edges: list[AlignmentEdge],
    orientations: dict[str, torch.Tensor],
) -> list[dict]:
    del tasks
    residuals = []
    for edge in edges:
        predicted = orientations[edge.source] @ orientations[edge.destination].T
        relative = (edge.q - predicted).norm() / max(edge.q.norm().item(), 1e-12)
        residuals.append(
            {
                "source_task": edge.source,
                "destination_task": edge.destination,
                "relative_frobenius": float(relative),
                "weight": edge.weight,
                "path": str(edge.path),
            }
        )
    return residuals


def synchronize_rotations(
    tasks: list[str], edges: list[AlignmentEdge], k: int
) -> tuple[dict[str, torch.Tensor], dict]:
    index = {task: offset for offset, task in enumerate(tasks)}
    block = torch.zeros(len(tasks) * k, len(tasks) * k)
    for edge in edges:
        i, j = index[edge.source], index[edge.destination]
        rows = slice(i * k, (i + 1) * k)
        cols = slice(j * k, (j + 1) * k)
        block[rows, cols] += float(edge.weight) * edge.q
        block[cols, rows] += float(edge.weight) * edge.q.T
    values, vectors = torch.linalg.eigh(block)
    order = torch.argsort(values, descending=True)
    top = vectors[:, order[:k]]
    orientations = {
        task: project_orthogonal(top[i * k : (i + 1) * k, :])
        for task, i in index.items()
    }
    residuals = rotation_residuals(tasks, edges, orientations)
    errors = [row["relative_frobenius"] for row in residuals]
    return orientations, {
        "top_eigenvalues": [float(value) for value in values[order[: min(k, len(values))]]],
        "rotation_residual_mean": _mean(errors),
        "rotation_residual_std": _sample_std(errors),
        "rotation_residuals": residuals,
    }


def synchronize_scales(
    tasks: list[str], edges: list[AlignmentEdge]
) -> tuple[dict[str, float], dict]:
    index = {task: offset for offset, task in enumerate(tasks)}
    rows, rhs = [], []
    for edge in edges:
        root_weight = float(edge.weight) ** 0.5
        row = torch.zeros(len(tasks))
        row[index[edge.source]] = -root_weight
        row[index[edge.destination]] = root_weight
        rows.append(row)
        rhs.append(root_weight * torch.log(torch.tensor(max(edge.alpha, 1e-12))).item())
    rows.append(torch.ones(len(tasks)))
    rhs.append(0.0)
    solution = torch.linalg.lstsq(torch.stack(rows), torch.tensor(rhs)).solution
    log_scales = {task: float(solution[index[task]]) for task in tasks}
    residuals = []
    for edge in edges:
        predicted = log_scales[edge.destination] - log_scales[edge.source]
        observed = float(torch.log(torch.tensor(max(edge.alpha, 1e-12))).item())
        residuals.append(
            {
                "source_task": edge.source,
                "destination_task": edge.destination,
                "observed_log_alpha": observed,
                "predicted_log_alpha": predicted,
                "absolute_error": abs(predicted - observed),
                "weight": edge.weight,
            }
        )
    errors = [row["absolute_error"] for row in residuals]
    return log_scales, {
        "scale_residual_mean_abs": _mean(errors),
        "scale_residual_std_abs": _sample_std(errors),
        "scale_residuals": residuals,
    }


def hub_map(
    source: str,
    destination: str,
    orientations: dict[str, torch.Tensor],
    log_scales: dict[str, float],
) -> tuple[torch.Tensor, float]:
    q = orientations[source] @ orientations[destination].T
    alpha = float(torch.exp(torch.tensor(log_scales[destination] - log_scales[source])).item())
    return q, alpha
