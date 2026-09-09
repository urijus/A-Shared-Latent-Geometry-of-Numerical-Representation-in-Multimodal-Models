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

For the baseline linear probe pipeline, create image datasets from the same
model-correct text JSONLs:

```bash
for modality in addition subtraction multiplication; do
  python -m src.data.images.generate_arithmetic_images \
    --input_jsonl "dataset/baseline/gemma4_12b_it/digits/model_correct_with_prompt/${modality}_baseline.jsonl" \
    --output_dir "dataset/images/gemma4_12b_it/digits/model_correct_with_prompt/${modality}"
done
```

Then extract multimodal activations:

```bash
python -m src.experiments.arithmetic_reference.linear_probes.text.extract_image_activations \
  --model gemma4_12b_it \
  --data_dir dataset/images/gemma4_12b_it/digits/model_correct_with_prompt \
  --output_dir outputs/activations/baseline_images/gemma4_12b_it/digits \
  --modalities addition subtraction multiplication \
  --prompt "Output ONLY a number. " \
  --position_strategy last_input \
  --batch_size 1
```

The saved activation files have the same shape convention as text activations:
`[examples, layers, positions, d_model]`. They currently save one position named
`last_input`, so the existing baseline probe runner can be reused by pointing
`--activation_dir` at the image activation directory.
