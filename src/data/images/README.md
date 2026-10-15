# Arithmetic Images

Render arithmetic samples as white-background PNG images with black centered
expressions such as `2 + 2 =`.

Generate a small image dataset from baseline arithmetic samples:

```bash
python -m src.data.images.generate_arithmetic_images \
  --operation all \
  --limit_per_operation 100 \
  --output_dir dataset/images
```

Convert an existing JSONL dataset while keeping its labels:

```bash
python -m src.data.images.generate_arithmetic_images \
  --input_jsonl dataset/baseline/addition_baseline.jsonl \
  --output_dir dataset/images/addition
```

Each output directory contains a `png/` folder and a metadata JSONL file. The
metadata preserves the source row and adds fields such as `image_text`,
`image_path`, `image_width`, `image_height`, `font_size`, `background_color`,
and `text_color`.

## Pipeline Use

For the main-paper reproduction, use the [experiment guide](../../../experiments/README.md#setup-and-data).
It specifies DejaVu Sans, 64 px, 384×384 images and the intersection of examples
answered correctly in both modalities. Rendering text-correct examples alone
does not perform the image correctness filter.

For a separate baseline probe exploration, create image datasets from existing
text JSONLs:

```bash
for modality in addition subtraction multiplication; do
  python -m src.data.images.generate_arithmetic_images \
    --input_jsonl "dataset/baseline/gemma4_12b_it/digits/model_correct_with_prompt/${modality}_baseline.jsonl" \
    --output_dir "dataset/images/gemma4_12b_it/digits/model_correct_with_prompt/${modality}"
done
```

Then extract multimodal activations:

```bash
python -m src.experiments.arithmetic_reference.linear_probes.image.extract_image_activations \
  --model gemma4_12b_it \
  --data_dir dataset/images/gemma4_12b_it/digits/model_correct_with_prompt \
  --output_dir outputs/activations/baseline_images/gemma4_12b_it/digits \
  --modalities addition subtraction multiplication \
  --prompt 'Output ONLY a number.' \
  --positions -1 --layers 43 \
  --batch_size 1
```

The saved activation files have the same shape convention as text activations:
`[examples, layers, positions, d_model]`. This command saves the final prompt
position (`-1`), so the baseline probe runner can be reused by pointing
`--activation_dir` at the image activation directory.
