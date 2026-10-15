# Reproducing the main experiments

After [installation](../README.md#installation), run the commands from the repository root, in the order below. Choose a new data
and output directory for each independent configuration.

| Group | Required inputs | Produces |
| --- | --- | --- |
| [Setup/data](#setup-and-data) | Local model and font | Matched text/image datasets |
| [DAS](#train-the-reference-spaces) | Paired data | Twelve trained bases and self/control evaluations |
| [Transfer](#causal-transfer-and-figure-2) | DAS bases | Direct/aligned transfer, controls and composition |
| [Readout/geometry](#readout-and-geometry-figure-3) | DAS bases and paired data | Ablations, exact D/L geometry and common frames |
| [Autoregressive](#autoregressive-transfer-figure-4) | DAS bases and readout audit | Latent swaps and next-digit evaluations |
| [Supporting checks](#supporting-checks) | Main maps and geometry | Repeat controls, overlap, 22D synchronization and successor |
| [Selection](#selection-checks) | Paired data | Layer, position and dimension sweeps |

## Setup and data

Use Gemma4-12B-it revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
Set the absolute path to a complete local snapshot and an installed DejaVu Sans
font below. Skip the symlink command if `models/gemma-4-12B-it` already points
to that snapshot.

```bash
set -euo pipefail
MODEL_SNAPSHOT='/absolute/path/to/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
FONT_PATH='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
test -f "$MODEL_SNAPSHOT/config.json"
test -f "$FONT_PATH"
mkdir -p models
ln -s "$MODEL_SNAPSHOT" models/gemma-4-12B-it

export DATA="$PWD/dataset/paper_reproduction"
export RUN="$PWD/results/paper_reproduction"
test ! -e "$DATA"
test ! -e "$RUN"
mkdir -p "$RUN/provenance"
git rev-parse HEAD > "$RUN/provenance/git_commit.txt"
git diff > "$RUN/provenance/uncommitted.diff"
python -m pip freeze > "$RUN/provenance/pip_freeze.txt"
nvidia-smi > "$RUN/provenance/nvidia_smi.txt"
sha256sum "$MODEL_SNAPSHOT/config.json" "$FONT_PATH" > "$RUN/provenance/input_hashes.txt"
```

The helper generates seed-0 arithmetic data, renders 64 px expressions on white
384×384 images, filters correctness in both modalities, and saves their
intersection. Prompt: `Output ONLY a number.`, thinking disabled, results 0–99.
The reference datasets contain 5049 addition and 5050 subtraction pairs; record
the counts in your generated manifest when comparing results.

```bash
python scripts/check_paired_datasets.py
python scripts/check_image_padding.py
python scripts/check_pca_seed.py
python scripts/check_cache_fingerprint.py

python scripts/prepare_main_datasets.py \
  --model gemma4_12b_it --font-path "$FONT_PATH" --output-root "$DATA"
cat "$DATA/paired/manifest.json"
sha256sum "$DATA/paired/"*.jsonl > "$RUN/provenance/paired_sha256.txt"
```

## Train the reference spaces

Train four conditions × seeds 0/1/2 at layer 43, final prompt token, rank 22.
The commands specify the 70/15/15 sample split, 4096/512/512 pair caps, PCA
initialization, and 15-epoch/patience-3 budget. Keep random controls enabled;
transfer normalization requires their scores.

```bash
AUDIT="$RUN/das"
for operation in addition subtraction; do
  for modality in text image; do
    position=17
    if [ "$modality" = image ]; then position=-1; fi
    for seed in 0 1 2; do
      python experiments/01_arithmetic_reference/das_audit.py \
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

```bash
export AUDIT RUN
python - <<'PY'
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

## Causal transfer and Figure 2

Run after training all twelve bases. Direct/aligned evaluations use nine seed
combinations and the same 128 ordered destination pairs. Composition uses
transition-matched pairs; the separate rank sweep truncates the transport map.

```bash
CAUSAL="$RUN/causal_transfer"
PROC="$RUN/procrustes"
python experiments/02_cross_condition_transfer/causal_transfer.py \
  --model gemma4_12b_it --audit_root "$AUDIT" --output_dir "$CAUSAL" \
  --seeds 0 1 2 --split_seed 0 --layer 43 --k 22 --hook resid_post \
  --text_position 17 --image_position=-1 --max_autoregressive_pairs 128

python experiments/02_cross_condition_transfer/align_subspaces.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --causal_transfer_rows "$CAUSAL/transfer_results.jsonl" --output_dir "$PROC" \
  --seeds 0 1 2 --layer 43 --k 22 --hook resid_post \
  --text_position 17 --image_position=-1 --max_autoregressive_pairs 128 \
  --variants basic random_orthogonal displacement centroid_displacement \
             scaled_displacement scaled_random_orthogonal alpha_identity \
             cross_operation_scaled

python -m src.experiments.cross_condition_transfer.procrustes.generalization_test.backfill_previous_controls \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --causal_transfer_rows "$CAUSAL/transfer_results.jsonl" --output_dir "$PROC" \
  --variants unrestricted_linear --seeds 0 1 2 \
  --max_autoregressive_pairs 128 --text_position 17 --image_position=-1

python experiments/02_cross_condition_transfer/factorized_paths.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --causal_transfer_rows "$CAUSAL/transfer_results.jsonl" \
  --output_dir "$PROC/factorized_paths" --seeds 0 1 2 \
  --max_autoregressive_pairs 128 --text_position 17 --image_position=-1

python experiments/02_cross_condition_transfer/rank_sweep.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --causal_transfer_rows "$CAUSAL/transfer_results.jsonl" \
  --output_dir "$PROC/rank_sweep" --seeds 0 1 2 --k 22 \
  --max_autoregressive_pairs 128 --text_position 17 --image_position=-1

python visualizations/main_paper/plot_procrustes_main_results.py \
  --causal_dir "$CAUSAL" --procrustes_dir "$PROC" \
  --factorized_summary "$PROC/factorized_paths/factorized_paths_summary.jsonl" \
  --output_dir "$RUN/figures/figure2"
```
The plot uses `basic` for its Q bar. Use the `displacement` result rows for
rotation-only transfer on numerical displacements. Native plot layouts may
differ from the manuscript panels.

## Readout and geometry (Figure 3)

Audit readout ablations, extract activations, then fit exact D=9/L=13 geometry
and common frames. Keep extraction at **batch 1**. Geometry fits on 80 numerical
values and evaluates 20 held-out values per split, with split seeds 0/1/2.
Only the digit basis from the projection helper is used downstream; keep
`$AUDIT` pointing to the trained main bases.

```bash
READOUT="$RUN/readout_audit"
python experiments/03_readout_latent_geometry/readout_audit.py \
  --model gemma4_12b_it --audit_root "$AUDIT" --output_dir "$READOUT" \
  --seeds 0 1 2 --layer 43 --k 22 --max_pairs 128 --batch_size 16 \
  --pca_max_samples 3584 --pca_seed 0 --random_controls 20
READOUT_PROJECTED="$RUN/readout_projected_bases"
python -m src.experiments.readout_latent_geometry.controls.readout_ablated_subspaces \
  --model gemma4_12b_it --audit_root "$AUDIT" --output_root "$READOUT_PROJECTED" \
  --conditions das_pca_initialized --seeds 0 1 2 --layer 43 --k 22 \
  --load_device cuda --text_position 17 --image_position=-1
READOUT_BASIS="$READOUT_PROJECTED/digit_readout_basis.pt"
test -f "$READOUT_BASIS"

# The extractors expect legacy filenames; stage the intersected rows without
# altering the canonical paired files. Absolute image paths survive staging.
export DATA
python - <<'PY'
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
python -m src.experiments.arithmetic_reference.linear_probes.text.extract_activations \
  --model gemma4_12b_it --data_dir "$DATA/activation_input_text" \
  --output_dir "$ACT_TEXT" --modalities addition subtraction \
  --positions 17 --layers 43 --batch_size 1 --use_chat_template
python -m src.experiments.arithmetic_reference.linear_probes.image.extract_image_activations \
  --model gemma4_12b_it --data_dir "$DATA/activation_input_image" \
  --output_dir "$ACT_IMAGE" --modalities addition subtraction \
  --positions -1 --layers 43 --batch_size 1 --prompt 'Output ONLY a number.'

GEOM="$RUN/value_geometry"
SYNC="$RUN/synchronization_L"
LFACTOR="$RUN/factorization_L"
python experiments/03_readout_latent_geometry/readout_free_geometry.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --output_dir "$GEOM" --seeds 0 1 2 --value_split_seeds 0 1 2 \
  --train_value_fraction 0.8 --readout_free_source project \
  --random_controls 20 --rsa_permutations 1000

python experiments/04_global_geometry/pairwise_geometry.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$GEOM" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --output_dir "$RUN/causal_L_geometry" --seeds 0 1 2 \
  --value_split_seeds 0 1 2 --m_readout 9 --latent_dim 13

python experiments/04_global_geometry/synchronize_frames.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$GEOM" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --output_dir "$SYNC" --seeds 0 1 2 --value_split_seeds 0 1 2 \
  --m_readout 9 --latent_dim 13

python experiments/04_global_geometry/factorized_paths.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$GEOM" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --old_das_summary "$PROC/factorized_paths/factorized_paths_summary.jsonl" \
  --output_dir "$LFACTOR" --seeds 0 1 2 \
  --value_split_seeds 0 1 2 --m_readout 9 --latent_dim 13

python -m src.experiments.global_geometry.modality_synchronization.modality_only_synchronization \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$GEOM" --sync_L_dir "$SYNC" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --output_dir "$RUN/modality_only_L" --seeds 0 1 2 \
  --value_split_seeds 0 1 2 --m_readout 9 --latent_dim 13

python visualizations/main_paper/plot_causal_L_shared_numerical_geometry.py \
  --input_dir "$RUN/causal_L_geometry" --output_dir "$RUN/figures/figure3_geometry"
python visualizations/main_paper/plot_exact_L_common_frame_synchronization.py \
  --input-dir "$SYNC" --factorization-dir "$LFACTOR" \
  --activation-dir-text "$ACT_TEXT" --activation-dir-image "$ACT_IMAGE" \
  --output-dir "$RUN/figures/figure3_synchronization"
```

## Autoregressive transfer (Figure 4)

Use explicit `--m_readout 9` for the D=9/L=13 split. Swap and next-step analyses
use separate triplet samples, seed 1729. Next-step scoring teacher-forces the
shared first digit and continues from the patched prompt KV-cache.

```bash
LATENT="$RUN/latent_swap"
NEXT="$RUN/next_step_readout"
python experiments/05_autoregressive_transfer/latent_identity_swap.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --readout_audit_dir "$READOUT" --output_dir "$LATENT" \
  --seeds 0 1 2 --layer 43 --k 22 --m_mode fixed --m_readout 9 \
  --text_position 17 --image_position=-1 --n_triplets 256 \
  --triplet_seed 1729 --batch_size 16 --activation_batch_size 16

python experiments/05_autoregressive_transfer/next_step_readout.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --readout_audit_dir "$READOUT" --output_dir "$NEXT" \
  --seeds 0 1 2 --layer 43 --k 22 --m_mode fixed --m_readout 9 \
  --text_position 17 --image_position=-1 --max_triplets 128 \
  --triplet_seed 1729

python visualizations/main_paper/plot_latent_identity_causal_swap.py \
  --input-dir "$LATENT" --output-dir "$RUN/figures/figure4_swap"
python visualizations/main_paper/plot_latent_to_autoregressive_readout.py \
  --input-dir "$NEXT" --output-dir "$RUN/figures/figure4_readout"
```

## Supporting checks

Run after the main transfer and geometry analyses. These checks reuse the
three-seed main bases, not the shorter-budget selection bases.

```bash
EXT="$RUN/main_claims"
mkdir -p "$EXT"
for condition in das_pca_initialized untrained_pca_initialized random_subspace_in_pca_span; do
  for seed in 0 1 2; do
    python -m src.experiments.readout_latent_geometry.controls.repeat_transfer \
      --model gemma4_12b_it \
      --subspace "$AUDIT/text/addition/$condition/split_0/seed_$seed/subspace.pt" \
      --output_dir "$EXT/digit_echo/$condition/seed_$seed" \
      --prompt_template 'Output ONLY a number. Repeat this number: {n} =' \
      --min_value 0 --max_value 9 --max_pairs 90 --pair_seed 0 \
      --position=-1 --layer 43 --hook resid_post --batch_size 16 \
      --max_new_tokens 8 --run_autoregressive
  done
done

patterns=()
for modality in text image; do
  for operation in addition subtraction; do
    for condition in das_pca_initialized untrained_pca_initialized random_subspace_in_pca_span; do
      patterns+=(--subspace_glob "$AUDIT/$modality/$operation/$condition/split_0/seed_*/subspace.pt")
    done
  done
done
python -m src.experiments.readout_latent_geometry.controls.unembedding_overlap \
  --model gemma4_12b_it --load_device cpu --max_token_number 9 \
  --output_dir "$EXT/unembedding_overlap" "${patterns[@]}"
```

The overlap comparison uses the analytic random residual-subspace expectation.
The CPU overlap audit loads model weights; allow sufficient host memory.

### Repeat-number geometry

Project repeat values 0–99 into the arithmetic L bases at the final prompt
token. Geometry uses teacher-forced correctness; the second command separately
checks greedy generation for exactly the target answer length and runs 200
one-example-per-value resamples over values 10–89. This answer-length check is
not the strict eight-token generation criterion used in main causal evaluation.

```bash
python experiments/05_autoregressive_transfer/repeat_geometry.py \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$RUN/causal_L_geometry" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --repeat_transfer_dir "$EXT/repeat_geometry" \
  --repeat_activation_path "$EXT/repeat_geometry/repeat_activations_layer43_resid_post.pt" \
  --output_dir "$EXT/repeat_geometry" --seeds 0 1 2 --value_split_seeds 0 1 2 \
  --layer 43 --k 22 --m_readout 9 --latent_dim 13 --hook resid_post \
  --text_position 17 --image_position=-1 --repeat_position=-1 \
  --min_value 0 --max_value 99 --batch_size 1 --accuracy_mode teacher_forced \
  --max_new_tokens 8 --n_shuffles 20 --rsa_permutations 1000

python -m src.experiments.autoregressive_transfer.repeat_noise_control \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --previous_geometry_dir "$RUN/causal_L_geometry" \
  --repeat_geometry_dir "$EXT/repeat_geometry" --repeat_transfer_dir "$EXT/repeat_geometry" \
  --repeat_activation_path "$EXT/repeat_geometry/repeat_activations_layer43_resid_post.pt" \
  --digit_readout_basis_path "$READOUT_BASIS" \
  --activation_dir_text "$ACT_TEXT" --activation_dir_image "$ACT_IMAGE" \
  --output_dir "$EXT/repeat_noise_control" --seeds 0 1 2 --value_split_seeds 0 1 2 \
  --layer 43 --k 22 --m_readout 9 --latent_dim 13 --hook resid_post \
  --text_position 17 --image_position=-1 --repeat_position=-1 \
  --n_replicates 200 --replicate_seed 1729 --bootstrap_samples 10000 --run_ar_check
```

### Full-22D causal synchronization and successor

Collect each relation's own direct matched-seed map from its recorded row path.
Different fits can share filenames; do not select maps by searching basenames.

```bash
export PROC EXT
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['EXT']) / 'factorized_maps' / 'maps'
root.mkdir(parents=True, exist_ok=True)
rows = [json.loads(line) for line in
        (Path(os.environ['PROC']) / 'factorized_paths/factorized_paths_results.jsonl').read_text().splitlines()]
paths = [Path(row['component_map_paths'][0]).resolve() for row in rows
         if row['transport'] == 'direct' and row['source_seed'] == row['destination_seed']]
assert len(paths) == 36 and len({p.name for p in paths}) == 36
for path in paths:
    assert path.is_file(), path
    link = root / path.name
    if not link.is_symlink():
        link.symlink_to(path)
    assert link.resolve() == path
PY

python -m src.experiments.global_geometry.synchronization.synchronization \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --causal_transfer_rows "$CAUSAL/transfer_results.jsonl" \
  --factorized_dir "$EXT/factorized_maps" \
  --output_dir "$EXT/full_das_synchronization/full_graph" \
  --seeds 0 1 2 --layer 43 --k 22 --hook resid_post --split_seed 0 \
  --text_position 17 --image_position=-1 --train_fraction 0.7 --validation_fraction 0.15 \
  --alignment_seed 0 --max_pair_bank 65536 --max_autoregressive_pairs 128 \
  --activation_batch_size 16 --max_new_tokens 8 --weight_metric mean_cosine \
  --transports pairwise_direct synchronized_hub

python -m src.experiments.cross_condition_transfer.equivariance.equivariance_registered_linear \
  --model gemma4_12b_it --audit_root "$AUDIT" \
  --sync_dir "$EXT/full_das_synchronization/full_graph" \
  --output_dir "$EXT/successor_transfer_22D" \
  --fit_task text:addition --seeds 0 1 2 --layer 43 --k 22 --hook resid_post \
  --split_seed 0 --text_position 17 --image_position=-1 \
  --train_fraction 0.7 --validation_fraction 0.15 \
  --value_split_seed 0 --value_test_fraction 0.2 --value_validation_fraction 0.2 \
  --models orthogonal identity random_orthogonal shuffled_scaled_orthogonal oracle_target_state \
  --geometry_steps 1 2 5 10 --causal_steps 1 --max_causal_pairs_per_domain 128 \
  --activation_batch_size 16 --max_new_tokens 8 --top_k 5
```

Synchronization produces 72 rows: twelve directions × three seeds × two
transports. The successor operator is fitted using T+ only, in **full 22D DAS**,
with its own 60/20/20 value split. It reuses the frozen synchronization.

## Selection checks

Use a separate **seed-0, eight-epoch/patience-2** budget. The commands train
42 unique DAS configurations, reusing the layer-43/rank-22/final-position bases.

### Probes and full-activation patching

Run after staging the paired activation inputs in the geometry section.
Probes cover all 48 layers; causal patching checks layers 41–45. Text position
sweeps cover positions 10–17 at layer 43.

```bash
SELECT="$RUN/selection"
for analysis in depth position; do
  layers=($(seq 1 48))
  positions=(17)
  if [ "$analysis" = position ]; then
    layers=(43)
    positions=(10 11 12 13 14 15 16 17)
  fi
  for modality in text image; do
    if [ "$analysis" = position ] && [ "$modality" = image ]; then continue; fi
    for operation in addition subtraction; do
      out="$SELECT/activations/$modality/$analysis"
      patch_layers=(41 42 43 44 45)
      if [ "$analysis" = position ]; then patch_layers=(43); fi
      patch_out="$SELECT/patching/$modality/$analysis"
      if [ "$modality" = text ]; then
        python -m src.experiments.arithmetic_reference.linear_probes.text.extract_activations \
          --model gemma4_12b_it --data_dir "$DATA/activation_input_text" \
          --output_dir "$out" --modalities "$operation" --positions "${positions[@]}" \
          --layers "${layers[@]}" --batch_size 1 --use_chat_template
        python -m src.experiments.arithmetic_reference.activation_patching.text.run_activation_patching \
          --model-name gemma4_12b_it --operation "$operation" \
          --data-path "$DATA/paired/${operation}_text.jsonl" --output-dir "$patch_out" \
          --layers "${patch_layers[@]}" --positions "${positions[@]}" \
          --hooks resid_post --layer-mode individual --n-pairs 128 --seed 0
      else
        python -m src.experiments.arithmetic_reference.linear_probes.image.extract_image_activations \
          --model gemma4_12b_it --data_dir "$DATA/activation_input_image" \
          --output_dir "$out" --modalities "$operation" --positions -1 \
          --layers "${layers[@]}" --batch_size 1 --prompt 'Output ONLY a number.'
        python -m src.experiments.arithmetic_reference.activation_patching.image.run_activation_patching \
          --model-name gemma4_12b_it --operation "$operation" \
          --data-path "$DATA/paired/${operation}_image.jsonl" --data-root "$DATA/paired" \
          --output-dir "$patch_out" --layers "${patch_layers[@]}" --positions -1 \
          --hooks resid_post --layer-mode individual --n-pairs 128 --seed 0 \
          --prompt 'Output ONLY a number.'
      fi
      mkdir -p "$SELECT/probes/$modality"
      python -m src.experiments.arithmetic_reference.linear_probes.text.run_probes \
        --activation_dir "$out" --output_path "$SELECT/probes/$modality/${operation}_${analysis}.jsonl" \
        --modalities "$operation" --targets result result_mod_10 --split_mode random \
        --val_fraction 0.2 --seed 0 --epochs 30 --probe_batch_size 4096 --lr 0.001
    done
  done
done
```

### DAS layers, positions and dimensions

```bash
SELECT="$RUN/selection"
selection_das() {
  local modality="$1" operation="$2" layer="$3" k="$4" position="$5" output="$6"
  python experiments/01_arithmetic_reference/das_audit.py \
    --model gemma4_12b_it --modality "$modality" --operation "$operation" \
    --data_path "$DATA/paired/${operation}_${modality}.jsonl" --data_root "$DATA/paired" \
    --output_dir "$output" --target result --layer "$layer" --hook resid_post \
    --position "$position" --k "$k" --seed 0 --split_seed 0 \
    --train_fraction 0.7 --validation_fraction 0.15 \
    --max_train_pairs 4096 --max_validation_pairs 512 --max_test_pairs 512 \
    --batch_size 16 --epochs 8 --patience 2 --learning_rate 0.0001 \
    --pca_max_samples 3584 --pca_seed 0 --pca_variance_threshold 0.9 \
    --max_autoregressive_pairs 128 --max_new_tokens 8 --das_only
}
for layer in 41 42 43 44 45; do
  for modality in text image; do
    position=17
    if [ "$modality" = image ]; then position=-1; fi
    for operation in addition subtraction; do
      selection_das "$modality" "$operation" "$layer" 22 "$position" "$SELECT/das_layers/layer_$layer"
    done
  done
done
for position in 10 11 12 13 14 15 16; do
  for operation in addition subtraction; do
    selection_das text "$operation" 43 22 "$position" "$SELECT/das_positions/position_$position"
  done
done
for k in 16 32; do
  for modality in text image; do
    position=17
    if [ "$modality" = image ]; then position=-1; fi
    for operation in addition subtraction; do
      selection_das "$modality" "$operation" 43 "$k" "$position" "$SELECT/das_dimensions/DAS_audit_k_$k"
    done
  done
done
# The layer-43 k=22 runs also provide the final-position and dimension references.
mkdir -p "$SELECT/das_positions" "$SELECT/das_dimensions"
ln -s "$SELECT/das_layers/layer_43" "$SELECT/das_positions/position_17"
ln -s "$SELECT/das_layers/layer_43" "$SELECT/das_dimensions/DAS_audit_k_22"
```

## Comparing results

Keep the model revision, data, font, seeds and evaluation settings fixed.
Preserve configurations, ordered pairs and summaries; fresh correctness
filtering can change the data pool. Compare the same metrics and denominators:
normalized causal recovery, complete-answer AR, teacher-forced scores and
held-out value retrieval are different measurements.

Reference results: direct/scaled cross-modal recovery **53.2/91.4%**, exact-L
retrieval **75.5%**, synchronized retrieval **74.8%**, latent answer redirection
**69.2%**, and next-digit redirection **77.6%**.
