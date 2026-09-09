"""Two-panel schematic of image and rendered multimodal inputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = (
    REPO_ROOT
    / "visualizations"
    / "examples"
    / "image_dataset_sample"
    / "addition"
    / "png"
    / "000000.png"
)
OUTPUT_DIR = REPO_ROOT / "visualizations" / "appendix" / "examples" / "image_dataset_sample"

BACKGROUND = "#fdfdfd"
PANEL_FILL = "#f7f5f3"
SPINE = "#b9c0c2"
TEXT = "#171717"
MUTED = "#6f777a"
ACCENT = "#8f4f59"
SECONDARY = "#5f9b92"
SOFT_ACCENT = "#ead8ce"
SOFT_SECONDARY = "#d8e6e3"
SOFT_PROMPT = "#f1e6de"
SOFT_TEMPLATE = "#e9edf0"


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": [
                "Palatino Linotype",
                "Georgia",
                "Cambria",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "figure.titlesize": 13,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
        }
    )


def setup_panel(ax, panel_label: str, title: str) -> None:
    ax.set_axis_off()
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.0,
        1.03,
        panel_label,
        ha="left",
        va="bottom",
        fontsize=11.2,
        fontweight="bold",
        color=ACCENT,
        transform=ax.transAxes,
    )
    ax.text(
        0.12,
        1.03,
        title,
        ha="left",
        va="bottom",
        fontsize=11.2,
        color=TEXT,
        transform=ax.transAxes,
    )


def draw_box(ax, xy, width, height, facecolor=PANEL_FILL, edgecolor=SPINE, linewidth=0.9):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    return box


def draw_token(ax, x, y, label, width, facecolor, edgecolor=None, fontsize=8.2):
    draw_box(
        ax,
        (x, y),
        width,
        0.115,
        facecolor=facecolor,
        edgecolor=edgecolor if edgecolor is not None else SPINE,
        linewidth=0.75,
    )
    ax.text(
        x + width / 2,
        y + 0.0575,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        family="DejaVu Sans Mono",
        color=TEXT,
    )


def draw_image_condition(ax) -> None:
    image = mpimg.imread(IMAGE_PATH)
    image_ax = ax.inset_axes([0.18, 0.31, 0.64, 0.64])
    image_ax.imshow(image, interpolation="nearest")
    image_ax.set_axis_off()
    for spine in image_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(SPINE)
        spine.set_linewidth(0.9)
    image_ax.add_patch(
        patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=image_ax.transAxes,
            linewidth=0.9,
            edgecolor=SPINE,
            facecolor="none",
        )
    )
    draw_box(ax, (0.12, 0.07), 0.76, 0.13, facecolor=PANEL_FILL)
    ax.text(
        0.50,
        0.13,
        "Output ONLY a number.",
        ha="center",
        va="center",
        fontsize=8.4,
        family="DejaVu Sans Mono",
        color=TEXT,
    )


def draw_sequence_schematic(ax) -> None:
    colors = {"template": SOFT_TEMPLATE, "image": SOFT_SECONDARY, "prompt": SOFT_PROMPT}
    labels = [
        ("Template", SOFT_TEMPLATE),
        ("Image tokens", SOFT_SECONDARY),
        ("Expression", SOFT_PROMPT),
    ]

    for idx, (label, color) in enumerate(labels):
        x = 0.08 + idx * 0.25
        ax.add_patch(
            patches.Rectangle(
                (x, 0.84),
                0.028,
                0.035,
                linewidth=0.6,
                edgecolor=SPINE,
                facecolor=color,
            )
        )
        ax.text(x + 0.038, 0.857, label, va="center", ha="left", fontsize=7.6, color=MUTED)

    def chip(x, y, width, label, kind, fontsize=7.2):
        draw_box(ax, (x, y), width, 0.105, facecolor=colors[kind], linewidth=0.75)
        ax.text(
            x + width / 2,
            y + 0.0525,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            family="DejaVu Sans Mono",
            color=TEXT,
        )

    chip(0.06, 0.58, 0.085, "<bos>", "template", fontsize=6.9)
    chip(0.155, 0.58, 0.115, "<|turn>", "template", fontsize=6.8)
    chip(0.28, 0.58, 0.075, "user", "template", fontsize=6.9)
    chip(0.365, 0.58, 0.235, "[IMAGE TOKENS]", "image", fontsize=6.6)
    chip(0.61, 0.58, 0.335, "Output ONLY a number.", "prompt", fontsize=6.55)

    chip(0.13, 0.37, 0.12, "<turn|>", "template", fontsize=6.8)
    chip(0.265, 0.37, 0.12, "<|turn>", "template", fontsize=6.8)
    chip(0.40, 0.37, 0.09, "model", "template", fontsize=6.9)
    chip(0.505, 0.37, 0.16, "<|channel>", "template", fontsize=6.45)
    chip(0.68, 0.37, 0.105, "thought", "template", fontsize=6.7)
    chip(0.80, 0.37, 0.16, "<channel|>", "template", fontsize=6.45)


def main() -> None:
    style_matplotlib()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.25, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.42], "wspace": 0.035},
    )
    fig.patch.set_facecolor(BACKGROUND)

    setup_panel(axes[0], "A", "Image condition")
    setup_panel(axes[1], "B", "Final rendered sequence")

    draw_image_condition(axes[0])
    draw_sequence_schematic(axes[1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output_path = OUTPUT_DIR / f"input_conditions_three_panel.{suffix}"
        fig.savefig(output_path, dpi=350, facecolor=fig.get_facecolor())
        print(f"Saved {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
