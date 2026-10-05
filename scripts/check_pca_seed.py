"""CPU reproducibility check for PCA span and pre-training DAS basis."""

import ast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def get_node(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


audit = ROOT / "src/experiments/arithmetic_reference/das_audit/audit_das.py"
das = ROOT / "src/interventions/das.py"
nodes = [get_node(audit, "pca_space_with_at_least_k"),
         get_node(das, "random_subspace_from_pca"), get_node(das, "DASSubspace")]
module = ast.Module(body=nodes, type_ignores=[])
scope = {"torch": torch, "nn": torch.nn}
exec(compile(ast.fix_missing_locations(module), str(audit), "exec"), scope)

generator = torch.Generator().manual_seed(21)
features = torch.randn(40, 8, generator=generator)


def initialize():
    pca = scope["pca_space_with_at_least_k"](features, 3, 0.9, 17)
    torch.manual_seed(4344)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4344)
    basis = scope["random_subspace_from_pca"](pca, 3)
    return pca, scope["DASSubspace"](8, 3, initial_basis=basis).basis()


first_span, first_basis = initialize()
torch.randn(1000)
second_span, second_basis = initialize()
assert torch.allclose(first_span @ first_span.T, second_span @ second_span.T, atol=1e-5)
assert torch.allclose(first_basis @ first_basis.T, second_basis @ second_basis.T, atol=1e-5)
print("PCA span and initial DAS basis reproduce with identical ordered features and seeds.")
