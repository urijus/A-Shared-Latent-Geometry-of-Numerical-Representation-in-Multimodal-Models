# Reproduction record: main transport figure (Sec. 3, Fig. 2)

Target: `visualizations/main_paper/plot_procrustes_main_results.py` → `main_four_panel_procrustes_results`, the data behind Fig. 2 (direct reuse 0.43 vs fitted αQ 0.93).
Branch `xavi/replicate` at `cbba3f6` ("Adjusting defaults"). No experiment code was modified.

## Setup

- 1× A100-SXM4-80GB, 10 CPUs. Python 3.12.3, torch 2.6.0+cu124, pinned `requirements.txt`, except **transformers 5.17.0** (see deviations), which also requires safetensors 0.8.0 and tokenizers 0.23.2:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-torch-cu124.txt
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python "transformers==5.17.0"
```

  The uncommitted `pyproject.toml` on this branch makes the pinned requirements installable with `uv sync`. Like `requirements.txt`, it still pins transformers 5.8.1, so the last command above is needed on top of it.
- Model: `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, linked where the loader looks first:

```bash
ln -sfn <hf-cache>/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 models/gemma-4-12B-it
```

## Commands, in order

`uv run --no-sync python` below equals `python` inside the repo's `.venv`. All commands run from the repo root.

```bash
# 1. Datasets (seed 0): 5050 addition, 5050 subtraction, 1000 multiplication
uv run python -m src.data.baseline.generate_datasets

# 2. Text correctness filter: 5050/5050 correct for both operations
for op in addition subtraction; do
  uv run --no-sync python -m src.models.diagnostics.test_model_accuracy --model gemma4_12b_it \
    --data_path dataset/baseline/${op}_baseline.jsonl \
    --output_path dataset/baseline/gemma4_12b_it/digits/model_correct_with_prompt/${op}_baseline.jsonl
done

# 3. Render images from the text-correct subsets, then filter by image correctness
#    (addition 5049/5050, subtraction 5050/5050)
for op in addition subtraction; do
  uv run --no-sync python -m src.data.images.generate_arithmetic_images \
    --input_jsonl dataset/baseline/gemma4_12b_it/digits/model_correct_with_prompt/${op}_baseline.jsonl \
    --output_dir dataset/baseline_images/gemma4_12b_it/digits/accuracy_with_prompt/${op} \
    --font_path /path/to/Arial.ttf
  uv run --no-sync python -m src.models.diagnostics.test_image_model_accuracy --model gemma4_12b_it \
    --data_path dataset/baseline_images/gemma4_12b_it/digits/accuracy_with_prompt/${op}/${op}_images.jsonl \
    --output_path dataset/baseline_images/gemma4_12b_it/digits/model_correct_with_prompt/${op}/${op}_images.jsonl \
    --prompt "Output ONLY a number."
done

# 4. DAS, 4 conditions x seeds 0-2 (defaults: k=22, layer 43, resid_post, split_seed 0,
#    output results/final_exps/DAS_audit_k_22). Text runs fit two at a time on 80 GB;
#    image runs need the GPU alone (training peaks near 58 GB).
DAS="--epochs 15 --patience 3 --max_test_pairs 512 --max_autoregressive_pairs 128 --pca_max_samples 3584"
for op in addition subtraction; do for s in 0 1 2; do
  uv run --no-sync python -m src.experiments.arithmetic_reference.das_audit.audit_das \
    --modality text  --operation $op --seed $s $DAS
  uv run --no-sync python -m src.experiments.arithmetic_reference.das_audit.audit_das \
    --modality image --operation $op --seed $s --position -1 $DAS
done; done

# 5. Causal transfer. A text-only pass ran first (--tasks text:addition text:subtraction);
#    the full run reuses those rows from its cache.
uv run --no-sync python experiments/02_cross_condition_transfer/causal_transfer.py

# 6. Procrustes variants, the A_lin control, composed paths, rank sweep, plot
uv run --no-sync python experiments/02_cross_condition_transfer/align_subspaces.py \
  --variants basic random_orthogonal scaled_displacement scaled_random_orthogonal alpha_identity
uv run --no-sync python -m src.experiments.cross_condition_transfer.procrustes.generalization_test.backfill_previous_controls \
  --variants unrestricted_linear
uv run --no-sync python experiments/02_cross_condition_transfer/factorized_paths.py
uv run --no-sync python -m src.experiments.cross_condition_transfer.procrustes.rank_sweep.rank_sweep \
  --ranks 1 2 4 8 16 22 --output_dir results/final_exps/procrustes/rank_sweep
uv run --no-sync python visualizations/main_paper/plot_procrustes_main_results.py
```

Status on Wed 30 Sep, 13:00: steps 1–3, all text DAS runs, 4 of 6 image DAS runs and the text-only pass of step 5 are complete. The remaining image runs and steps 5–6 run through `logs/run_remaining.sh`.

## Deviations from repo defaults

| Setting | Repo default | Used | Reason |
|---|---|---|---|
| transformers | 5.8.1 | 5.17.0 | The current checkpoint declares `model_type: gemma4_unified`, which 5.8.1 cannot load |
| DAS epochs / patience | 8 / 2 | 15 / 3 | Appendix A: final k=22 subspaces use 15 epochs, patience 3 |
| DAS test / AR pairs | 128 / 32 | 512 / 128 | Appendix A: 512 test and 128 autoregressive test pairs |
| DAS PCA samples | 1024 | 3584 | Appendix A: PCA from up to 3584 training examples |
| Image DAS position | `17` | `-1` | `17` is the text semantic position; causal transfer and procrustes expect `-1` for images |
| Image filter prompt | "Solve the arithmetic expression in the image. Output ONLY a number." | "Output ONLY a number." | The prompt in the paper and in the DAS image code |
| Image font | first of DejaVuSans, Arial | Arial | The paper's example image `000000` is Arial; DejaVu renders a footed "1" and wider digits |
| Image dataset path | `dataset/images/...` (images README) | `dataset/baseline_images/...` | The path the DAS defaults read |
| Rank sweep output | `results/paper/procrustes/rank_sweep` | `results/final_exps/procrustes/rank_sweep` | The folder the plot reads |
| Rank sweep ranks | 1..22 | 1 2 4 8 16 22 | Saves about 4 GPU hours; only the wide figure variant uses it |

Transfer and procrustes stages keep their default of 32 autoregressive pairs, since the paper does not state a count.

## Results so far vs the paper

DAS self-intervention AR IIA (128 held-out pairs; sampling error about ±0.04):

| Condition | Seeds 0 / 1 / 2 | Mean | Paper (Appendix C diagonal) |
|---|---|---|---|
| T+ | 0.852 / 0.844 / 0.828 | 0.84 | 0.75 |
| T− | 0.875 / 0.844 / 0.875 | 0.865 | 0.96 |
| I+ | 0.750 / 0.688 / 0.711 | 0.72 | 0.55 |
| I− | 0.727 / pending / pending | — | 0.84 |

The random PCA-span control scores 0.000 in every run. Addition subspaces are stronger than the paper's in both modalities and subtraction subspaces weaker, by 0.09–0.17. That exceeds sampling noise; the transformers and checkpoint change is the prime suspect.

Text half of the direct-reuse matrix (3×3 seed pairs, 32 pairs per evaluation):

| Cell | Raw AR IIA (ours / paper) | Normalized τ (ours / paper) |
|---|---|---|
| T+ → T− | 0.69 / 0.84 | 0.80 / 0.88 |
| T− → T+ | 0.65 / 0.60 | 0.77 / 0.81 |

Normalized diagonals are 0.98 (T+) and 1.04 (T−). The largest gap, T+ → T− raw at −0.15, is under 2 binomial standard errors (about ±0.08 per raw cell at 32 pairs).

## Issues for the authors

- Data generation, both correctness filters, `audit_das.py`, the A_lin backfill and the rank sweep have no public commands.
- The image pipeline is CPU-bound: the processor reruns on every image each epoch, so one image DAS run takes about 7 h on an A100 and the GPU idles about 60% of the time. Caching processor outputs should roughly halve that without changing results (estimated, not tested).
- Full-vector patching at the final position scores 0.09–0.10 (text) and 0.00 (image) autoregressive IIA, far below the 22-dimensional DAS subspace. This may deserve a sentence in the paper.
