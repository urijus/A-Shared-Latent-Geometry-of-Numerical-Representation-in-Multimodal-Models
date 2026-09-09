# Readout-Overlap Audits

Auxiliary analyses for testing whether learned DAS spaces are explained by
current digit-readout directions.

| Module | Purpose |
| --- | --- |
| `unembedding_overlap.py` | Measure overlap between DAS bases and digit-token unembedding directions. |
| `readout_ablated_subspaces.py` | Project the centered digit-readout span out of saved DAS bases. |
| `self_ablated_eval.py` | Re-evaluate arithmetic IIA before and after readout ablation. |
| `readout_transition_sweep.py` | Combine readout overlap, repeat transfer, and self-IIA across layers. |
| `repeat_transfer.py` | Test whether a DAS space behaves like a generic number-repeat output channel. |

The paper-facing entry points for the corresponding main analyses are in
`experiments/03_readout_latent_geometry/` and
`experiments/05_autoregressive_transfer/`.
