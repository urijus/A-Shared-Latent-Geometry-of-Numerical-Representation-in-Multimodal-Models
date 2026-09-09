"""Summarize Ministral Stage 1 DAS sweep outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from src.common import load_jsonl, save_jsonl


TASK_LABELS = {
    ("text", "addition"): "T+",
    ("text", "subtraction"): "T-",
    ("image", "addition"): "I+",
    ("image", "subtraction"): "I-",
}
TASK_ORDER = ["T+", "T-", "I+", "I-"]


def task_label(row: dict) -> str:
    return TASK_LABELS.get(
        (row.get("modality"), row.get("operation")),
        f"{row.get('modality')}:{row.get('operation')}",
    )


def result_paths(results_root: Path, phase: str) -> list[Path]:
    phase_root = results_root / phase
    return sorted(phase_root.glob("**/das_pca_initialized/**/results.jsonl"))


def load_rows(results_root: Path, phase: str) -> list[dict]:
    rows = []
    for path in result_paths(results_root, phase):
        for row in load_jsonl(path):
            if not isinstance(row, dict):
                continue
            rows.append({**row, "results_path": str(path)})
    return rows


def fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        if rows
        else len(str(header))
        for index, header in enumerate(headers)
    ]
    lines = []
    header_line = " | ".join(
        str(header).ljust(widths[index]) for index, header in enumerate(headers)
    )
    lines.append(header_line)
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            " | ".join(
                str(value).ljust(widths[index]) for index, value in enumerate(row)
            )
        )
    for line in lines:
        print(line)
    return lines


def phase_a_summary(rows: list[dict]) -> tuple[list[str], dict]:
    by_layer_task = {(int(row["layer"]), task_label(row)): row for row in rows}
    layers = sorted({layer for layer, _task in by_layer_task})
    table_rows = []
    for layer in layers:
        t = by_layer_task.get((layer, "T+"), {})
        i = by_layer_task.get((layer, "I+"), {})
        table_rows.append(
            [
                layer,
                fmt(t.get("variable_teacher_forced_iia")),
                fmt(t.get("autoregressive_iia")),
                fmt(i.get("variable_teacher_forced_iia")),
                fmt(i.get("autoregressive_iia")),
            ]
        )
    lines = print_table(
        ["layer", "T+ TF-IIA", "T+ AR-IIA", "I+ TF-IIA", "I+ AR-IIA"],
        table_rows,
    )
    scored = []
    for layer in layers:
        values = [
            by_layer_task[(layer, task)].get("autoregressive_iia")
            for task in ("T+", "I+")
            if (layer, task) in by_layer_task
        ]
        if len(values) == 2 and all(value is not None for value in values):
            scored.append((min(values), mean(values), layer))
    suggested = [layer for _minimum, _avg, layer in sorted(scored, reverse=True)[:2]]
    if suggested:
        print(f"Suggested Phase B layers: {suggested}")
    return lines, {"suggested_phase_b_layers": suggested}


def phase_b_summary(rows: list[dict]) -> tuple[list[str], dict]:
    by_layer_task = {(int(row["layer"]), task_label(row)): row for row in rows}
    layers = sorted({layer for layer, _task in by_layer_task})
    table_rows = []
    scored = []
    for layer in layers:
        ar_values = []
        row = [layer]
        for task in TASK_ORDER:
            item = by_layer_task.get((layer, task), {})
            tf = item.get("variable_teacher_forced_iia")
            ar = item.get("autoregressive_iia")
            row.extend([fmt(tf), fmt(ar)])
            if ar is not None:
                ar_values.append(ar)
        table_rows.append(row)
        if len(ar_values) == 4:
            scored.append((min(ar_values), mean(ar_values), -int(layer == 40), layer))
    headers = ["layer"]
    for task in TASK_ORDER:
        headers.extend([f"{task} TF", f"{task} AR"])
    lines = print_table(headers, table_rows)
    selected = sorted(scored, reverse=True)[0][-1] if scored else None
    if selected is not None:
        print(f"Suggested common layer: {selected}")
    return lines, {"suggested_common_layer": selected}


def phase_c_summary(rows: list[dict]) -> tuple[list[str], dict]:
    by_task_k = {(task_label(row), int(row["k"])): row for row in rows}
    ks = sorted({k for _task, k in by_task_k})
    table_rows = []
    for k in ks:
        t = by_task_k.get(("T+", k), {})
        i = by_task_k.get(("I+", k), {})
        table_rows.append(
            [
                k,
                fmt(t.get("variable_teacher_forced_iia")),
                fmt(t.get("autoregressive_iia")),
                fmt(i.get("variable_teacher_forced_iia")),
                fmt(i.get("autoregressive_iia")),
            ]
        )
    lines = print_table(
        ["k", "T+ TF-IIA", "T+ AR-IIA", "I+ TF-IIA", "I+ AR-IIA"],
        table_rows,
    )
    selected = None
    if 32 in ks:
        t_ref = by_task_k.get(("T+", 32), {}).get("autoregressive_iia")
        i_ref = by_task_k.get(("I+", 32), {}).get("autoregressive_iia")
        if t_ref is not None and i_ref is not None:
            for k in ks:
                t_ar = by_task_k.get(("T+", k), {}).get("autoregressive_iia")
                i_ar = by_task_k.get(("I+", k), {}).get("autoregressive_iia")
                if t_ar is None or i_ar is None:
                    continue
                if t_ar >= 0.95 * t_ref and i_ar >= 0.95 * i_ref:
                    selected = k
                    break
    if selected is not None:
        print(f"Suggested selected k: {selected}")
    return lines, {"suggested_k": selected}


def phase_d_summary(rows: list[dict]) -> tuple[list[str], dict]:
    by_task = defaultdict(list)
    for row in rows:
        by_task[task_label(row)].append(row)
    table_rows = []
    for task in TASK_ORDER:
        task_rows = by_task.get(task, [])
        ar = [row.get("autoregressive_iia") for row in task_rows]
        tf = [row.get("variable_teacher_forced_iia") for row in task_rows]
        ar = [value for value in ar if value is not None]
        tf = [value for value in tf if value is not None]
        table_rows.append(
            [
                task,
                len(task_rows),
                fmt(mean(ar)) if ar else "NA",
                fmt(stdev(ar)) if len(ar) > 1 else "0.000" if ar else "NA",
                fmt(mean(tf)) if tf else "NA",
                fmt(stdev(tf)) if len(tf) > 1 else "0.000" if tf else "NA",
            ]
        )
    lines = print_table(
        ["task", "runs", "AR mean", "AR std", "TF mean", "TF std"],
        table_rows,
    )
    return lines, {}


def summarize(rows: list[dict], phase: str) -> tuple[list[str], dict]:
    if phase == "phase_a":
        return phase_a_summary(rows)
    if phase == "phase_b":
        return phase_b_summary(rows)
    if phase in {"phase_c", "phase_c_confirm"}:
        return phase_c_summary(rows)
    if phase == "phase_d":
        return phase_d_summary(rows)
    headers = ["phase", "task", "layer", "k", "seed", "TF-IIA", "AR-IIA"]
    table_rows = [
        [
            phase,
            task_label(row),
            row.get("layer"),
            row.get("k"),
            row.get("seed"),
            fmt(row.get("variable_teacher_forced_iia")),
            fmt(row.get("autoregressive_iia")),
        ]
        for row in rows
    ]
    return print_table(headers, table_rows), {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("results/mistral/stage1"))
    parser.add_argument("--phase", required=True)
    parser.add_argument("--selected_layer", type=int)
    parser.add_argument("--selected_k", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_root, args.phase)
    if not rows:
        print(f"No DAS result rows found for {args.phase} under {args.results_root}")
        return
    print(f"Loaded {len(rows)} rows for {args.phase}")
    lines, metadata = summarize(rows, args.phase)
    metadata = {
        **metadata,
        "phase": args.phase,
        "results_root": str(args.results_root),
        "selected_layer": args.selected_layer,
        "selected_k": args.selected_k,
        "n_rows": len(rows),
    }

    summary_dir = args.results_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, summary_dir / f"{args.phase}_rows.jsonl")
    (summary_dir / f"{args.phase}_table.txt").write_text(
        "\n".join(lines) + "\n" + json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved summary rows/table under: {summary_dir}")


if __name__ == "__main__":
    main()
