# Paper–reproduction discrepancy audit and corrected main-paper rerun

This file records what the discrepancy investigation established and gives a **new, common-setup rerun** for the main results in Sections 4–6 of *Causal Numerical Representation*. It is a runbook, not a claim that the commands below have already been executed or that their results equal the published numbers. The attached `REPRODUCTION.md` documents the earlier reproduction; its numerical results and commands refer to a different setup. Run every block below from the repository root, in order, on a Linux machine with the Gemma checkpoint and a GPU capable of loading it. Use a **new** output root for each attempt.

## 1. What the comparison showed

The question was why `REPRODUCTION.md` showed stronger addition and weaker subtraction than the submission paper, and whether that weakened the paper's causal-transport result. We audited code and saved results, recovered and replayed historical subspaces on fixed pairs, and separately varied image batch size, PCA cap, padding correction, and intervention position. These GPU comparisons used **seed 0**; the paper reports three-seed averages.

| Source of mismatch | Finding |
| --- | --- |
| Evaluation | Some reproduction comparisons used 32 pairs, while the paper's main causal evaluation used 128. Self, control, and transfer must use the **same ordered destination pairs** for a normalized score. |
| Image training and PCA | Image batch size 2 versus 16 and PCA cap 1,024 versus 3,584 changed the learned subspaces. Four current seed-0 subspaces differ from archived ones. |
| Subtraction | The appropriate 128-pair paper reference is 88.8% text / 71.9% image, versus 86.5% / 72.7% in the reproduction. Much of the apparent gap came from comparing unlike references. |
| Image addition | On identical historical pairs, 64.1% became 68.0% with batch 16, 64.8% with larger PCA, and 70.3% with both. On new pairs, both changes gave 71.9%, close to the reproduction's 71.6%. Larger PCA did not consistently improve text addition. |
| Image padding | Correcting per-row left-padding offsets changed image subtraction 69.5% → 72.7% on historical pairs, but 72.7% → 71.9% on new pairs. Addition's training pool has no mixed answer lengths, so this bug does not explain its increase. |
| Intervention position | Moving subspaces trained on the final prompt token one token earlier gave 0%. Training *at* that earlier token remains untested. |
| Saved transport | The saved headline direct-to-scaled scores, 0.430 → 0.933, recompute from saved rows. Their underlying model executions have **not** been verified by a fresh transport run. |

The exact contributions of pair sampling, training variability, changed seed-0 files, model/software versions, and regenerated images remain unresolved, especially for text addition. The single-seed, unmatched-environment checks do not quantify how much of the **overall** discrepancy is explained. For a targeted causal check before broad retraining, replay all three archived seeds on identical pairs in both environments, including fresh direct/scaled transport and matched self/control scores. The pipeline below then tests the corrected setup end to end, including the image-padding fix.

## 2. Experiment map: main manuscript to repository

The section numbers here are those in the **PDF**, not the historical labels in `experiments/*/README.md`. Every downstream row consumes the corrected DAS checkpoints; do not mix their outputs with archived or reproduction bases.

| PDF result | Entrypoint(s) | Inputs / key settings | Main artifact |
| --- | --- | --- | --- |
| §4, self efficacy and direct cross-condition reuse | `experiments/01_arithmetic_reference/das_audit.py`; `experiments/02_cross_condition_transfer/causal_transfer.py` | Four conditions × seeds 0,1,2; layer 43, `resid_post`, rank 22; 128 matched held-out pairs and PCA-span random control | DAS `results.jsonl`, `subspace.pt`; `destination_normalized_transfer_matrix.jsonl` |
| §4, scaled orthogonal alignment, controls and composition, Fig. 2 | `experiments/02_cross_condition_transfer/align_subspaces.py`, `factorized_paths.py`, `rank_sweep.py`; module `src.experiments.cross_condition_transfer.procrustes.generalization_test.backfill_previous_controls`; `visualizations/main_paper/plot_procrustes_main_results.py` | Reuse those bases and direct-transfer rows; fit maps on separate alignment data; evaluate matching destination pairs; all ranks 1–22 for the plotted sweep | Variant matrices, `factorized_paths_summary.jsonl`, `rank_sweep_summary.jsonl`, Fig. 2 |
| §5, digit readout overlap and causal ablations | `experiments/03_readout_latent_geometry/readout_audit.py`; module `src.experiments.readout_latent_geometry.controls.readout_ablated_subspaces` | Same four tasks/seeds; centered digit-unembedding span of rank 9; 128 evaluation pairs; random rank-matched controls | Ablation tables; `digit_readout_basis.pt` from the projection helper |
| §5, held-out numerical geometry and exact 13D latent space, Fig. 3a–b | Text/image activation extractors (below); `experiments/03_readout_latent_geometry/readout_free_geometry.py`; `experiments/04_global_geometry/pairwise_geometry.py` | Intersected samples, layer 43, final prompt token; D=9/L=13; value split seeds 0,1,2 (80/20 values), random/permutation controls | `value_splits.json`, L geometry summaries, Fig. 3a–b inputs |
| §5, one global frame and withheld relation/condition, Fig. 3c | `experiments/04_global_geometry/synchronize_frames.py`, `factorized_paths.py`; module `src.experiments.global_geometry.modality_synchronization.modality_only_synchronization`; `visualizations/main_paper/plot_exact_L_common_frame_synchronization.py` | Reuse the exact same value splits, activations, digit basis and four-condition bases | `L_synchronization_summary.json`, `L_factorization_summary.json`, Fig. 3c |
| §6, matched/mismatched D+L and delayed readout, Fig. 4 | `experiments/05_autoregressive_transfer/latent_identity_swap.py`, `next_step_readout.py`; corresponding `visualizations/main_paper` plot scripts | Same bases; D=9/L=13; controlled triplets; prompt-position intervention and subsequent KV-cache continuation | Per-condition swap and next-step tables, Fig. 4 |

The `unrestricted_linear`/\(A_{\rm lin}\) control is **not** an `align_subspaces.py` variant. Its dedicated module command below generates it directly in the fresh run's Procrustes directory. The rank sweep is required by the current Fig. 2 plotting script.

## 3. Definitive setup for this rerun

| Item | Corrected choice |
| --- | --- |
| Model | `google/gemma-4-12B-it`, revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; verify the locally loaded checkpoint before running. |
| Data | Generate arithmetic examples with seed 0; keep only `sample_id`s answered correctly in **both** text and image; preserve identical order. Images: Arial, 384×384, font 64, prompt `Output ONLY a number.`, thinking disabled. |
| Split and pair selection | Sample-disjoint 70/15/15 train/validation/test, `split_seed=0`; unique-pair sampler seeds 0/1/2 for train/validation/test; caps 4,096/512/512 pairs. The **first 128 ordered saved test pairs**, sampled by seed 2, are the fixed main causal set for each operation in both modalities and all DAS seeds. They are seeded sampled pairs, not the first arithmetic expressions in numerical order. Stop if a task cannot supply 128. |
| DAS | Layer 43 `resid_post`, final prompt token (text raw position `17`, image `-1`); rank `k=22`; seeds 0,1,2; batch 16; PCA cap 3,584, variance threshold 0.90, `pca_seed=0`; Adam learning rate 1e-4, up to 15 epochs, patience 3; answer target `result`; max 8 generated tokens. |
| Loss | Training code minimizes **sum of target-answer token negative log probabilities per example, averaged over examples**. A two-token answer therefore contributes two token losses; this is not answer-length-normalized NLL. The paper's description must be revised if the fresh run keeps this objective. |
| Evaluation | 128 fixed held-out pairs for the headline autoregressive causal comparisons; self/control/transfer on the same destination pair IDs and order. Teacher-forced scores can use the saved 512 test pairs where the script supports it; record each actual denominator. |
| Image indexing | Convert unpadded intervention/donor/answer indices with each row's offset from the **actual** model encoding (`padded_width - attention_mask.sum()` for left padding; zero for right padding). Patch the final prompt token before the answer; score answer token `j` from logits at `j-1`. |
| Caches and records | Keep a fresh run directory. Transfer/Procrustes cache fingerprints bind code, model/processor, dataset and ordered pairs, both basis hashes, intervention settings, generation settings, maps and normalization inputs. Missing/mismatched fingerprints are rejected or recomputed. Record git/environment, 12 configs, histories, basis hashes and actual PCA sample counts. |

The 128-pair rule applies to the **main causal evaluation**. Section 5's value-heldout retrieval uses a different unit: 20 candidate values per value split. Section 6 uses seeded triplets (the swap script defaults to 256; next-step tracing defaults to 128) and must report those denominators separately rather than calling them 128 pairs.

### Differences from the submitted setup and from the reproduction

| Item | Submission paper | Corrected rerun / interpretation |
| --- | --- | --- |
| Correctness intersection | Appendix A.1 already says matched cross-modal comparisons use the intersection of correct examples. | The new dataset helper **implements and records** that intersection for all four main condition datasets. The earlier reproduction did not consistently do so. This implements the paper's stated intent rather than changing its rule. |
| Batch, PCA and pair count | Appendix A.3 already specifies batch 16, PCA cap 3,584 and 128 autoregressive pairs. | Use exactly those explicit flags for text **and** image. The reproduction's image batch 2, PCA cap 1,024 and some 32-pair evaluations were deviations from the paper. |
| Sampling reproducibility | The paper gives sample/pair fractions and three DAS seeds, but not all concrete RNG seeds, ordered pair files or hashes. | Fix dataset generation seed 0, split seed 0, pair sampler seeds 0/1/2, `pca_seed=0`, DAS seeds 0/1/2; save and verify the first 128 ordered test pair IDs. These are new details to report, not a change to the mathematical design. |
| DAS loss | Appendix A.3 describes answer-length-normalized teacher-forced NLL. | The actual `result` training code sums target-token log probabilities per example, then averages examples. Document this **paper–code discrepancy** and the objective used for the fresh results. |
| Image padding | The paper assumes the final prompt token is patched and answer tokens are scored in order; it does not describe padding offsets. | Correct each padded row's indices in training, validation and batched causal evaluation. This can change results and should be noted alongside the fresh figures. |
| Transport normalization/cache | The paper defines destination self/control normalization but does not specify cache identity or matched ordered pair manifests. | Evaluate self/control/transfer on the same destination rows; reject stale caches with fingerprints. Recompute the published transport claim rather than reusing archived aggregates. |
| Exact D/L in §6 | §5–6 describe the exact D=9/L=13 decomposition. | Pass `--m_readout 9` explicitly to the latent-swap script, whose default is 5. If the archived Fig. 4 used 5, its numbers and rank wording need reconciliation. |
| Software and images | The paper does not uniquely pin all package, processor, font and rendered-image file hashes. | Record the actual environment, snapshot, font and new dataset hashes; regenerated images/environment may prevent numerical identity with archived outputs. |

## 4. Commands: corrected fresh run

Commands use Bash and the repository's `experiments/...` entrypoints. They deliberately use one isolated data root and one isolated result root. Replace only the font path and the model snapshot path with real local paths. Do not point either at historical result directories.

### 4.1 Environment and provenance

```bash
set -euo pipefail
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-torch-cu124.txt
uv pip install --python .venv/bin/python -r requirements.txt

MODEL_SNAPSHOT='<hf-cache>/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
FONT_PATH='/absolute/path/to/Arial.ttf'
test -f "$MODEL_SNAPSHOT/config.json"
test -f "$FONT_PATH"
mkdir -p models
ln -sfnT "$MODEL_SNAPSHOT" models/gemma-4-12B-it

DATA=dataset/corrected_128
RUN=results/corrected_128_fresh_01
test ! -e "$DATA"                 # remove this guard only if resuming the exact same inputs
test ! -e "$RUN"                  # choose a new name for a new run
mkdir -p "$RUN/provenance"
git rev-parse HEAD > "$RUN/provenance/git_commit.txt"
git status --short > "$RUN/provenance/git_status.txt"
git diff > "$RUN/provenance/uncommitted.diff"
uv pip freeze --python .venv/bin/python > "$RUN/provenance/pip_freeze.txt"
nvidia-smi > "$RUN/provenance/nvidia_smi.txt"
printf '%s\n' "$MODEL_SNAPSHOT" > "$RUN/provenance/model_snapshot.txt"
sha256sum "$MODEL_SNAPSHOT/config.json" > "$RUN/provenance/model_config_sha256.txt"
sha256sum "$FONT_PATH" > "$RUN/provenance/font_sha256.txt"
```

The reproduction used an A100-SXM4-80GB, Python 3.12.3 and PyTorch 2.6.0+cu124; record the **actual** rerun versions above. A different checkpoint or processor revision changes the experiment. `requirements.txt` now pins the Transformers stack; use its checked-in versions instead of silently upgrading packages.

### 4.2 Generate and intersect correctness-filtered datasets

```bash
uv run --no-sync python scripts/check_paired_datasets.py
uv run --no-sync python scripts/check_image_padding.py
uv run --no-sync python scripts/check_pca_seed.py
uv run --no-sync python scripts/check_cache_fingerprint.py

uv run --no-sync python scripts/prepare_main_datasets.py \
  --model gemma4_12b_it --font-path "$FONT_PATH" --output-root "$DATA"
cat "$DATA/paired/manifest.json"
sha256sum "$DATA/paired/"*.jsonl > "$RUN/provenance/paired_sha256.txt"
```

The helper runs generation, text correctness filtering, image rendering, image correctness filtering, and **intersection** in that order. Its paired files are `"$DATA/paired/{addition,subtraction}_{text,image}.jsonl"`; `manifest.json` records counts. Image paths in the paired image rows are relative to `"$DATA/paired"`. If those filtered inputs already exist and only the intersection needs rebuilding, use `--only-intersect --output-root "$DATA"` instead of regenerating images. Do not use text-correct rows alone for image training.

### 4.3 Train and audit the twelve main DAS spaces

```bash
AUDIT="$RUN/das"
for operation in addition subtraction; do
  for modality in text image; do
    position=17
    if [ "$modality" = image ]; then position=-1; fi
    for seed in 0 1 2; do
      uv run --no-sync python experiments/01_arithmetic_reference/das_audit.py \
        --model gemma4_12b_it --modality "$modality" --operation "$operation" \
        --data_path "$DATA/paired/${operation}_${modality}.jsonl" \
        --data_root "$DATA/paired" --output_dir "$AUDIT" \
        --target result --layer 43 --hook resid_post --position "$position" \
        --k 22 --seed "$seed" --split_seed 0 \
        --train_fraction 0.7 --validation_fraction 0.15 \
        --max_train_pairs 4096 --max_validation_pairs 512 --max_test_pairs 512 \
        --batch_size 16 --epochs 15 --patience 3 --learning_rate 0.0001 \
        --pca_max_samples 3584 --pca_seed 0 --pca_variance_threshold 0.9 \
        --max_autoregressive_pairs 128 --max_new_tokens 8
    done
  done
done
```

Leave `--das_only` **unset**: direct transport needs the saved random-PCA-span control as well as DAS self scores. For each task/seed, inspect `"$AUDIT/$modality/$operation/das_pca_initialized/split_0/seed_$seed/"`: `config.json`, `training_history.jsonl`, `results.jsonl` (including `basis_sha256` and actual `pca_sample_count`), `heldout_pairs.jsonl`, and `subspace.pt`. Check for all 12 directories before proceeding. The same task's pair IDs must agree across seeds, and the same operation's text/image IDs must agree; the dataset preflight tests the sampler on a small fixture, while the saved rows are the actual run evidence.

```bash
export AUDIT RUN
uv run --no-sync python - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['AUDIT'])
pair_hashes = {}
for operation in ('addition', 'subtraction'):
    reference = None
    for modality in ('text', 'image'):
        for seed in (0, 1, 2):
            folder = root / modality / operation / 'das_pca_initialized' / 'split_0' / f'seed_{seed}'
            for name in ('config.json', 'training_history.jsonl', 'results.jsonl', 'heldout_pairs.jsonl', 'subspace.pt'):
                assert (folder / name).is_file(), folder / name
            result = json.loads((folder / 'results.jsonl').read_text().splitlines()[0])
            assert result.get('basis_sha256') and result.get('pca_sample_count'), folder
            rows = [json.loads(line) for line in (folder / 'heldout_pairs.jsonl').read_text().splitlines()]
            assert len(rows) >= 128, (folder, len(rows))
            pair_ids = [(r['base_sample_id'], r['source_sample_id']) for r in rows[:128]]
            if reference is None:
                reference = pair_ids
            assert pair_ids == reference, (operation, modality, seed)
    pair_hashes[operation] = hashlib.sha256(
        json.dumps(reference, separators=(',', ':')).encode('utf-8')).hexdigest()
    print(operation, '128 ordered held-out pair IDs agree across modalities and seeds')
Path(os.environ['RUN'], 'provenance', 'ordered_128_pair_hashes.json').write_text(
    json.dumps(pair_hashes, indent=2) + '\n', encoding='utf-8')
PY
```

### 4.4 Section 4: direct transfer, alignments, controls and Fig. 2

```bash
CAUSAL="$RUN/causal_transfer"
PROC="$RUN/procrustes"
uv run --no-sync python experiments/02_cross_condition_transfer/causal_transfer.py \
  --model gemma4_12b_it --audit_root "$AUDIT" --output_dir "$CAUSAL" \
  --seeds 0 1 2 --split_seed 0 --layer 43 --k 22 --hook resid_post \
  --text_position 17 --image_position=-1 --max_autoregressive_pairs 128

uv run --no-sync python experiments/02_cross_condition_transfer/align_subspaces.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --causal_transfer_rows "$CAUSAL/transfer_results.jsonl" --output_dir "$PROC" \
  --seeds 0 1 2 --layer 43 --k 22 --hook resid_post \
  --text_position 17 --image_position=-1 --max_autoregressive_pairs 128 \
  --variants basic random_orthogonal displacement centroid_displacement \
             scaled_displacement scaled_random_orthogonal alpha_identity \
             cross_operation_scaled

uv run --no-sync python -m src.experiments.cross_condition_transfer.procrustes.generalization_test.backfill_previous_controls \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --causal_transfer_rows "$CAUSAL/transfer_results.jsonl" --output_dir "$PROC" \
  --variants unrestricted_linear --seeds 0 1 2 \
  --max_autoregressive_pairs 128 --text_position 17 --image_position=-1

uv run --no-sync python experiments/02_cross_condition_transfer/factorized_paths.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --causal_transfer_rows "$CAUSAL/transfer_results.jsonl" \
  --output_dir "$PROC/factorized_paths" --seeds 0 1 2 \
  --max_autoregressive_pairs 128 --text_position 17 --image_position=-1

uv run --no-sync python experiments/02_cross_condition_transfer/rank_sweep.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --causal_transfer_rows "$CAUSAL/transfer_results.jsonl" \
  --output_dir "$PROC/rank_sweep" --seeds 0 1 2 --k 22 \
  --max_autoregressive_pairs 128 --text_position 17 --image_position=-1

uv run --no-sync python visualizations/main_paper/plot_procrustes_main_results.py \
  --causal_dir "$CAUSAL" --procrustes_dir "$PROC" \
  --factorized_summary "$PROC/factorized_paths/factorized_paths_summary.jsonl" \
  --output_dir "$RUN/figures/figure2"
```

The rank command's default is the full 1–22 sweep. Fit samples and evaluation pairs remain separate. Inspect row fingerprints and the **ordered** destination pair IDs for direct, scaled, random, \(A_{\rm lin}\), and self/control normalization; task/seed labels alone do not establish comparability. The current plot expects the unrestricted-linear and rank-sweep artifacts, which is why both commands appear above.

### 4.5 Section 5: readout basis, held-out values and Fig. 3

```bash
READOUT="$RUN/readout_audit"
uv run --no-sync python experiments/03_readout_latent_geometry/readout_audit.py \
  --model gemma4_12b_it --audit_root "$AUDIT" --output_dir "$READOUT" \
  --seeds 0 1 2 --layer 43 --k 22 --max_pairs 128 --batch_size 16 \
  --pca_max_samples 3584 --pca_seed 0 --random_controls 20
READOUT_PROJECTED="$RUN/readout_projected_bases"
uv run --no-sync python -m src.experiments.readout_latent_geometry.controls.readout_ablated_subspaces \
  --model gemma4_12b_it --audit_root "$AUDIT" --output_root "$READOUT_PROJECTED" \
  --conditions das_pca_initialized --seeds 0 1 2 --layer 43 --k 22 \
  --load_device cuda --text_position 17 --image_position=-1
READOUT_BASIS="$READOUT_PROJECTED/digit_readout_basis.pt"
test -f "$READOUT_BASIS"

# The extractors expect legacy filenames; stage the intersected rows without
# altering the canonical paired files. Absolute image paths survive staging.
export DATA
uv run --no-sync python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['DATA']).resolve()
paired = root / 'paired'
text_dir = root / 'activation_input_text'
image_dir = root / 'activation_input_image'
text_dir.mkdir(exist_ok=True)
for operation in ('addition', 'subtraction'):
    (text_dir / f'{operation}_baseline.jsonl').write_bytes(
        (paired / f'{operation}_text.jsonl').read_bytes())
    target = image_dir / operation
    target.mkdir(parents=True, exist_ok=True)
    with (paired / f'{operation}_image.jsonl').open(encoding='utf-8') as src, \
         (target / f'{operation}_images.jsonl').open('w', encoding='utf-8') as dst:
        for line in src:
            row = json.loads(line)
            row['image_path'] = str((paired / row['image_path']).resolve())
            dst.write(json.dumps(row) + '\n')
PY

ACT_TEXT="$RUN/activations/text"
ACT_IMAGE="$RUN/activations/image"
uv run --no-sync python -m src.experiments.arithmetic_reference.linear_probes.text.extract_activations \
  --model gemma4_12b_it --data_dir "$DATA/activation_input_text" \
  --output_dir "$ACT_TEXT" --modalities addition subtraction \
  --positions 17 --layers 43 --batch_size 1 --use_chat_template
uv run --no-sync python -m src.experiments.arithmetic_reference.linear_probes.image.extract_image_activations \
  --model gemma4_12b_it --data_dir "$DATA/activation_input_image" \
  --output_dir "$ACT_IMAGE" --modalities addition subtraction \
  --positions -1 --layers 43 --batch_size 1 --prompt 'Output ONLY a number.'

GEOM="$RUN/value_geometry"
SYNC="$RUN/synchronization_L"
LFACTOR="$RUN/factorization_L"
uv run --no-sync python experiments/03_readout_latent_geometry/readout_free_geometry.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --output_dir "$GEOM" --seeds 0 1 2 --value_split_seeds 0 1 2 \
  --train_value_fraction 0.8 --readout_free_source project \
  --random_controls 20 --rsa_permutations 1000

uv run --no-sync python experiments/04_global_geometry/pairwise_geometry.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$GEOM" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --output_dir "$RUN/causal_L_geometry" --seeds 0 1 2 \
  --value_split_seeds 0 1 2 --m_readout 9 --latent_dim 13

uv run --no-sync python experiments/04_global_geometry/synchronize_frames.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$GEOM" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --output_dir "$SYNC" --seeds 0 1 2 --value_split_seeds 0 1 2 \
  --m_readout 9 --latent_dim 13

uv run --no-sync python experiments/04_global_geometry/factorized_paths.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$GEOM" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --old_das_summary "$PROC/factorized_paths/factorized_paths_summary.jsonl" \
  --output_dir "$LFACTOR" --seeds 0 1 2 \
  --value_split_seeds 0 1 2 --m_readout 9 --latent_dim 13

uv run --no-sync python -m src.experiments.global_geometry.modality_synchronization.modality_only_synchronization \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$GEOM" --sync_L_dir "$SYNC" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --output_dir "$RUN/modality_only_L" --seeds 0 1 2 \
  --value_split_seeds 0 1 2 --m_readout 9 --latent_dim 13

uv run --no-sync python visualizations/main_paper/plot_causal_L_shared_numerical_geometry.py \
  --input_dir "$RUN/causal_L_geometry" --output_dir "$RUN/figures/figure3_geometry"
uv run --no-sync python visualizations/main_paper/plot_exact_L_common_frame_synchronization.py \
  --input-dir "$SYNC" --factorization-dir "$LFACTOR" \
  --activation-dir-text "$ACT_TEXT" --activation-dir-image "$ACT_IMAGE" \
  --output-dir "$RUN/figures/figure3_synchronization"
```

The activation extractors resolve prompt positions without each row's left-padding offset. Running **feature extraction** at `--batch_size 1` avoids mixed-width padding and keeps the selected token correct. This does **not** change DAS training's batch 16; the image DAS path uses the corrected row-wise indexing. Confirm that the saved activation `position_names` are `17` and `-1`, respectively, and that labels carry the same ordered sample IDs as the paired data. The geometry and synchronization scripts reuse the same saved `value_splits.json` rather than inventing new held-out values.

`readout_audit.py` writes the causal ablation tables but does not itself serialize the digit basis. The projection helper writes `digit_readout_basis.pt`; downstream commands use **only that basis file** from its output root. Its projected checkpoint copies are not substitutes for the freshly trained, evaluated DAS checkpoints in `"$AUDIT"`.

### 4.6 Section 6: latent identity, continuation and Fig. 4

```bash
LATENT="$RUN/latent_swap"
NEXT="$RUN/next_step_readout"
uv run --no-sync python experiments/05_autoregressive_transfer/latent_identity_swap.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --readout_audit_dir "$READOUT" --output_dir "$LATENT" \
  --seeds 0 1 2 --layer 43 --k 22 --m_mode fixed --m_readout 9 \
  --text_position 17 --image_position=-1 --n_triplets 256 \
  --triplet_seed 1729 --batch_size 16 --activation_batch_size 16

uv run --no-sync python experiments/05_autoregressive_transfer/next_step_readout.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --readout_audit_dir "$READOUT" --output_dir "$NEXT" \
  --seeds 0 1 2 --layer 43 --k 22 --m_mode fixed --m_readout 9 \
  --text_position 17 --image_position=-1 --max_triplets 128 \
  --triplet_seed 1729

uv run --no-sync python visualizations/main_paper/plot_latent_identity_causal_swap.py \
  --input-dir "$LATENT" --output-dir "$RUN/figures/figure4_swap"
uv run --no-sync python visualizations/main_paper/plot_latent_to_autoregressive_readout.py \
  --input-dir "$NEXT" --output-dir "$RUN/figures/figure4_readout"
```

The manuscript's §5 exact decomposition uses rank 9 for D and rank 13 for L. `latent_identity_swap.py` still has a **default** `m_readout=5`; the explicit `9` above makes the §6 test consistent with the manuscript's stated exact split. This choice may change Fig. 4 relative to archived results and must be reported, not silently treated as a numerical reproduction. Both experiments use fixed triplet seeds and should report realized eligible triplet counts for every task/seed.

## 5. What must change in the paper after the rerun

These are **planned text/table updates**, conditional on the fresh outputs; the published numbers above should not be replaced before evaluating them.

| Paper statement/location | Correction to document from the fresh run |
| --- | --- |
| Methods and Appendix dataset construction | State both correctness filters and the text–image `sample_id` intersection, image-rendering settings, exact model/processor revision, dataset counts and ordered-pair hashes. Historical experiments that used another pool must be labelled as such. |
| DAS setup in Methods/Appendix | Give per-modality batch 16, PCA cap 3,584 and actual PCA sample counts, `pca_seed=0`, train seeds 0–2, `split_seed=0`, pair-sampler seeds 0/1/2, sample split 70/15/15, pair caps 4,096/512/512, 15 epochs/patience 3, layer/hook/positions. Record checkpoint basis hashes and training histories. |
| DAS objective equation/text | Replace any “length-normalized answer NLL” claim with the implemented sum of target-token losses, averaged across examples, unless the code and entire experiment are explicitly changed and rerun. |
| §4 Fig. 2, its caption and Appendix transport settings | Specify 128 **matched** destination pairs for self, random control and transfer; map fitting/evaluation splits; all three seeds; direct/scaled/control variants and normalization. Replace the saved 0.430/0.933 only with fresh verified values. |
| §5 Fig. 3 / Appendix geometry | Identify the rank-9 digit-readout span and exact rank-13 L, the shared value splits (80/20, seeds 0–2), activation extraction at the last prompt token, random/permutation controls, and realized candidate/value counts. |
| §6 Fig. 4 / Appendix continuation | State explicit D/L ranks, triplet seed and realized counts, matched/mismatched donor construction, first-token and next-token scoring, and whether `m_readout=9` was used. Reconcile this with any archived run using the script default of 5. |
| Limitations or discrepancy note | If discussing full-vector patching, label its 0.09–0.10 text / 0.00 image AR IIA as an observed control. |

Finally, compare fresh outputs with the paper **at the same pair count, sample set, seeds, denominator and metric**. Preserve the old `REPRODUCTION.md` and old results as historical evidence. A successful run produces a new internally comparable result set; it does not guarantee bitwise reproduction of the submission's original figures.