import argparse
import math
import random
from pathlib import Path

import torch

from src.common.io import load_jsonl, save_jsonl
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.probes.fourier import (
    batches,
    load_probe_grid,
    prompts_for_samples,
    resolve_positions,
    steer_hidden,
)
from src.models import (
    get_blocks,
    load_hf_model,
    resolve_model_for_loading,
    validate_block_layers,
)


def build_pairs(samples, target, periods, n_pairs, seed):
    rng = random.Random(seed)
    bases = rng.sample(samples, min(n_pairs, len(samples)))
    pairs = []
    for pair_id, base in enumerate(bases):
        sources = [
            source for source in samples
            if any(
                int(source[target]) % period != int(base[target]) % period
                for period in periods
            )
        ]
        if sources:
            pairs.append(
                {"pair_id": pair_id, "base": base, "source": rng.choice(sources)}
            )
    return pairs


def coordinate_metrics(before, after, target, probe, period, alpha):
    weight = probe["weight"].to(before.device)
    bias = probe["bias"].to(before.device)
    before_xy = before.float() @ weight.T + bias
    after_xy = after.float() @ weight.T + bias
    radius = before_xy.norm(dim=-1)
    angle = target.float() * (2 * math.pi / period)
    target_xy = alpha * radius[:, None] * torch.stack(
        (torch.cos(angle), torch.sin(angle)), dim=-1
    )
    before_phase = torch.atan2(before_xy[:, 1], before_xy[:, 0]) % (2 * math.pi)
    after_phase = torch.atan2(after_xy[:, 1], after_xy[:, 0]) % (2 * math.pi)
    return {
        "before_xy": before_xy,
        "after_xy": after_xy,
        "target_xy": target_xy,
        "distance_to_target": (after_xy - target_xy).norm(dim=-1),
        "distance_to_base": (after_xy - before_xy).norm(dim=-1),
        "before_phase": before_phase,
        "after_phase": after_phase,
        "before_value": before_phase * period / (2 * math.pi),
        "after_value": after_phase * period / (2 * math.pi),
    }


@torch.no_grad()
def run(args, model, tokenizer, model_name):
    layers = validate_block_layers(model, args.layers)
    samples = load_jsonl(args.data_path)
    pairs = build_pairs(samples, args.target, args.periods, args.n_pairs, args.seed)
    probes, probe_paths = load_probe_grid(
        args.probe_root,
        args.modality,
        args.target,
        args.periods,
        layers,
        args.positions,
        args.method,
    )
    rows = []

    for pair_batch in batches(pairs, args.batch_size):
        base_samples = [pair["base"] for pair in pair_batch]
        targets = torch.tensor(
            [int(pair["source"][args.target]) for pair in pair_batch],
            device=model.device,
        )
        prompts = prompts_for_samples(tokenizer, base_samples, args.use_chat_template)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoding = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        hf_positions = torch.tensor(
            [resolve_positions(tokenizer, prompt, args.positions) for prompt in prompts],
            device=model.device,
        )
        batch_indices = torch.arange(len(pair_batch), device=model.device)

        def make_hook(layer):
            def hook(_module, _inputs, output):
                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                for position_index, position in enumerate(args.positions):
                    token_positions = hf_positions[:, position_index]
                    before = hidden[batch_indices, token_positions]
                    after = steer_hidden(
                        before,
                        targets,
                        probes[layer][position],
                        args.periods,
                        args.alpha,
                    )
                    hidden[batch_indices, token_positions] = after

                    for period_index, period in enumerate(args.periods):
                        metrics = coordinate_metrics(
                            before,
                            after,
                            targets,
                            probes[layer][position][period_index],
                            period,
                            args.alpha,
                        )
                        for index, pair in enumerate(pair_batch):
                            rows.append(
                                {
                                    "model": model_name,
                                    "modality": args.modality,
                                    "target": args.target,
                                    "pair_id": pair["pair_id"],
                                    "base_sample_id": pair["base"].get("sample_id"),
                                    "source_sample_id": pair["source"].get("sample_id"),
                                    "base_value": int(pair["base"][args.target]),
                                    "source_value": int(pair["source"][args.target]),
                                    "layer": layer,
                                    "position": position,
                                    "period": period,
                                    "alpha": args.alpha,
                                    "before_xy": metrics["before_xy"][index].cpu().tolist(),
                                    "after_xy": metrics["after_xy"][index].cpu().tolist(),
                                    "target_xy": metrics["target_xy"][index].cpu().tolist(),
                                    "distance_to_target": float(metrics["distance_to_target"][index]),
                                    "distance_to_base": float(metrics["distance_to_base"][index]),
                                    "before_phase": float(metrics["before_phase"][index]),
                                    "after_phase": float(metrics["after_phase"][index]),
                                    "before_decoded_value": float(metrics["before_value"][index]),
                                    "after_decoded_value": float(metrics["after_value"][index]),
                                }
                            )
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

            return hook

        handles = [
            get_blocks(model)[layer - 1].register_forward_hook(make_hook(layer))
            for layer in layers
        ]
        try:
            model(**encoding, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

    for row in rows:
        row["probe_paths"] = {
            str(layer): {
                str(position): [str(path) for path in paths]
                for position, paths in positions.items()
            }
            for layer, positions in probe_paths.items()
        }
    save_jsonl(rows, args.output_path)
    print(f"Saved {len(rows)} coordinate checks: {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--probe_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--modality", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--periods", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--method", choices=["gd", "ridge"], default="ridge")
    parser.add_argument("--n_pairs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_chat_template", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_chat_template = args.use_chat_template or uses_chat_template(args.model)
    model_path, model_name = resolve_model_for_loading(args.model)
    model, tokenizer = load_hf_model(model_path)
    model.eval()
    run(args, model, tokenizer, model_name)


if __name__ == "__main__":
    main()
