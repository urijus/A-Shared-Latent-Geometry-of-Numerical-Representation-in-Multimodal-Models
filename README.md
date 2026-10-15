# A Shared Latent Geometry of Numerical Representation in Multimodal Models

<p align="center">
  <b>Do multimodal models represent the same numerical result using the same internal geometry?</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key Results</a> •
  <a href="#repository-map">Code</a> •
  <a href="#installation">Installation</a> •
  <a href="#reproducing-the-experiments">Reproduction</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  📄 Paper coming soon &nbsp;&nbsp;•&nbsp;&nbsp; 🧾 arXiv coming soon
</p>

---

## Overview

This repository contains the research code accompanying **A Shared Latent Geometry of Numerical Representation in Multimodal Models**.

We study numerical representations in `Gemma4-12B-it` using matched addition and subtraction problems presented as text or rendered images. This lets us change the computational route and input modality while holding the numerical result fixed.

The experiments distinguish five questions:

1. **Accessibility:** can numerical variables be decoded from the residual stream?
2. **Causal relevance:** can swapping a low-dimensional state change the generated answer?
3. **Causal reuse:** does a state learned in one condition remain effective in another?
4. **Shared geometry:** does numerical structure persist after removing current digit-readout coupling?
5. **Autoregressive use:** does that latent identity subsequently control later generated digits?

The central finding is that **shared numerical representation does not require identical neural coordinates**. Condition-specific causal spaces are related by simple transformations, and their readout-orthogonal components preserve a common numerical geometry. Interventions on latent identity redirect later digits while largely preserving the immediate digit.

## Paper

**A Shared Latent Geometry of Numerical Representation in Multimodal Models**<br>
**Oriol Juan Sabater**

- **Paper / arXiv:** coming soon.
- **Thesis:** Master's Thesis in Computer Vision, September 2026.
- **Code:** this repository; citation metadata will be updated with the public manuscript.

---

## Key Results

### 1. Arithmetic results have strong late causal representations

Distributed Alignment Search (DAS) learns numerical subspaces whose interventions replace the generated result. Complete-answer autoregressive accuracy is **83.1% / 86.2% / 74.7% / 69.3%** across the four conditions:

| Symbol | Operation | Input |
| --- | --- | --- |
| `T+` | Addition | Text |
| `T−` | Subtraction | Text |
| `I+` | Addition | Image |
| `I−` | Subtraction | Image |

The main experiments use **layer 43** and **k=22** as a common reference across conditions.

→ [Arithmetic reference experiments](experiments/01_arithmetic_reference/)

### 2. Operations share causal directions more directly than modalities

Direct reuse across operations within one modality remains strong. Changing modality creates a larger coordinate mismatch: direct same-operation cross-modal reuse recovers **53.2%** of the destination-trained causal effect. Across all eight cross-modal relations, including operation changes, direct recovery averages **45.5%**.

### 3. Rotation and scale recover most cross-modal causal efficacy

Numerical displacements are aligned using

```math
A_{j \leftarrow i} = \alpha Q, \qquad Q^\top Q = I.
```

Across the four same-operation cross-modal directions, **rotation alone recovers 71.8%**, rising to **91.4%** with a fitted global scale. An unrestricted linear map reaches **96.3%**. These are destination-normalized recovery scores, rather than raw answer accuracies.

Composing modality and operation transformations retains approximately **79.9%** recovery when both factors change. The fitted maps describe representational correspondence; they do not establish that the model computes these transformations internally.

→ [Cross-condition transfer experiments](experiments/02_cross_condition_transfer/)

### 4. Numerical geometry survives outside the current digit readout

Inside the chosen rank-22 DAS space, coupling to the centered digit-unembedding span defines an exact decomposition

```math
R = D \oplus L,
```

with **9 readout-coupled dimensions D** and **13 readout-orthogonal dimensions L**. Removing digit-readout coupling almost eliminates immediate answer control, while exact L retains:

| Held-out geometry metric | Exact L | Random readout-orthogonal 13D control |
| --- | ---: | ---: |
| No-fit RSA | **0.765** | 0.545 |
| Top-1 value retrieval | **75.5%** | 38.3% |

Exact L has mean transition cosine **0.807**; shuffled correspondences give approximately **5.2%** top-1 retrieval. Evaluation values are excluded from alignment fitting, with 20 held-out candidates per split.

→ [Readout and latent geometry experiments](experiments/03_readout_latent_geometry/)

### 5. The latent spaces admit a common numerical frame

One orientation and scale per condition can replace independent pairwise maps while preserving held-out geometry:

| Mapping | Transition cosine | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| Independent pairwise | 0.801 | 73.4% | 98.4% |
| Full synchronization | 0.791 | 74.8% | 98.9% |
| Leave-one-relation-out | 0.762 | 71.1% | 98.3% |

These comparisons use matched DAS seeds; the 75.5% pairwise result above uses all nine source/destination seed combinations. A modality-only frame reaches **18.1%** top-1, supporting separate condition frames. Common coordinates describe the relational geometry, rather than a unique physical subspace shared literally by every condition.

### 6. Modality and operation structure predicts a held-out condition

Predicting a missing corner of the text/image × addition/subtraction design reaches **0.629** transition cosine and **58.9%** top-1 retrieval. Shuffled correspondence gives **−0.109** cosine. Prediction uses the other three conditions; an evaluation-only anchor map places the true held-out condition in the reference frame for comparison.

→ [Global geometry experiments](experiments/04_global_geometry/)

### 7. Latent numerical identity is used during later generation

For two-digit values sharing a first digit, replacing only L redirects complete-answer generation toward the alternative value in **69.2%** of trials. After teacher-forcing the shared first digit and continuing with the patched prompt KV-cache, the alternative next digit is selected in **77.6%** of trials, with **no further intervention**.

The induced change becomes aligned with digit-readout directions downstream. D and L are an analytical decomposition of a learned space, rather than two demonstrated neural modules.

→ [Autoregressive transfer experiments](experiments/05_autoregressive_transfer/)

---

## Repository Map

| Paper | Question | Entrypoints | Implementation |
| --- | --- | --- | --- |
| §3 | Where and in what dimension is the causal result state? | [01_arithmetic_reference](experiments/01_arithmetic_reference/) | [arithmetic_reference](src/experiments/arithmetic_reference/) |
| §4 / Fig. 2 | How do causal spaces transfer and align? | [02_cross_condition_transfer](experiments/02_cross_condition_transfer/) | [cross_condition_transfer](src/experiments/cross_condition_transfer/) |
| §5 / Fig. 3a–b | What survives outside digit readout? | [03_readout_latent_geometry](experiments/03_readout_latent_geometry/) | [readout_latent_geometry](src/experiments/readout_latent_geometry/) |
| §5 / Fig. 3c | Is there a consistent common frame? | [04_global_geometry](experiments/04_global_geometry/) | [global_geometry](src/experiments/global_geometry/) |
| §6 / Fig. 4 | Does latent identity control subsequent digits? | [05_autoregressive_transfer](experiments/05_autoregressive_transfer/) | [autoregressive_transfer](src/experiments/autoregressive_transfer/) |

Other useful entry points:

- **Run order, exact commands and settings:** [experiment guide](experiments/README.md).
- **Plots:** [main-paper figure generators](visualizations/main_paper/).
- **Model aliases and local loading:** [src/models/hf.py](src/models/hf.py).
- **Image generation:** [image data guide](src/data/images/README.md).
- **Supplementary analyses:** [src/experiments/supplementary/](src/experiments/supplementary/).
- **Geometry and interventions:** [src/geometry/](src/geometry/) and [src/interventions/](src/interventions/).

---

## Installation

The reference results were obtained with **Linux, Python 3.10.20, PyTorch 2.11.0+cu128 and Transformers 5.17.0**, using an **RTX PRO 6000 Blackwell Max-Q GPU (approximately 96 GB)**. [requirements.txt](requirements.txt) pins this software stack, including the CUDA 12.8 PyTorch build used with that GPU.

```bash
git clone https://github.com/urijus/A-Shared-Latent-Geometry-of-Numerical-Representation-in-Multimodal-Models.git
cd A-Shared-Latent-Geometry-of-Numerical-Representation-in-Multimodal-Models
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation before starting GPU experiments:

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

<details>
<summary><b>Different GPU, CPU-only analysis, or installation problems?</b></summary>

The supplied requirements target the tested Linux/CUDA environment. To use another platform, choose compatible PyTorch/torchvision wheels using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) and adapt the two PyTorch entries and wheel index in a local copy of requirements.txt. The [official version table](https://pytorch.org/get-started/previous-versions/) lists the reference 2.11.0 / 0.26.0 CUDA 12.8 pair. Keep the other experiment dependencies pinned and record deviations.

If CUDA is unavailable, check the selected wheels, NVIDIA driver and GPU visibility before running model inference. CPU analysis can consume saved activations/results; the full reproduction includes GPU model inference and DAS training. The approximately 96 GB device is the hardware used, not a measured minimum-memory requirement.

</details>

Model access requires the appropriate Hugging Face permissions and license acceptance. For reproducible fingerprinted experiments, use a local checkpoint of revision **707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7**, linked under `models/gemma-4-12B-it`; the [experiment guide](experiments/README.md#setup-and-data) shows how.

---

## Reproducing the Experiments

Start with the [experiment guide](experiments/README.md). It contains runnable Bash command blocks using the existing Python entrypoints, with explicit inputs, output paths and dependencies. No Slurm installation or cluster launchers are required.

| Run in order | Reproduces |
| --- | --- |
| [Setup and data](experiments/README.md#setup-and-data) | Pinned model, deterministic rendering, correctness intersection and shared pairs |
| [Train DAS](experiments/README.md#train-the-reference-spaces) | Four conditions × three seeds; reference causal spaces |
| [Causal transfer](experiments/README.md#causal-transfer-and-figure-2) | Direct/aligned transfer, controls, composition and map-rank sweep |
| [Readout and geometry](experiments/README.md#readout-and-geometry-figure-3) | Ablations, exact D/L geometry, synchronization and held-out tests |
| [Autoregressive transfer](experiments/README.md#autoregressive-transfer-figure-4) | Latent swaps, next-digit control and delayed readout |
| [Supporting checks](experiments/README.md#supporting-checks) | Digit echo, repeat numbers, overlap, 22D synchronization and successor |
| [Selection checks](experiments/README.md#selection-checks) | Layer/position localization and k=16/22/32 sensitivity |

The main setup uses **layer 43, final prompt token, resid_post, k=22**, seeds **0/1/2**, batch **16**, up to **15 epochs** with patience **3**, and **128 ordered AR test pairs**. Selection checks use **seed 0, eight epochs and patience two**. The guide documents their separate budgets and scoring criteria.

For an individual command's options:

```bash
python experiments/01_arithmetic_reference/das_audit.py --help
```

The wrappers under `experiments/` delegate to scientific implementations under `src/experiments/`. Dataset generation, filtering, training and evaluation must all use one consistent run configuration. Fresh filtering or different hardware/software may change numerical results.

Model weights, datasets and generated outputs are excluded from Git. The commands write to your chosen `dataset/` and `results/` directories.

## Validation

```bash
python scripts/check_paired_datasets.py
python scripts/check_image_padding.py
python scripts/check_pca_seed.py
python scripts/check_cache_fingerprint.py
python scripts/validate_repository.py
```

These checks exercise input consistency, padding, PCA seeding, cache validity and public CLI entrypoints. They do not rerun the scientific GPU experiments.

---

## Citation

Citation metadata will be updated with the public manuscript. For now:

```bibtex
@misc{juan2026sharedlatentgeometry,
  title  = {A Shared Latent Geometry of Numerical Representation in Multimodal Models},
  author = {Juan Sabater, Oriol},
  year   = {2026},
  note   = {Master's Thesis in Computer Vision. Manuscript forthcoming.}
}
```

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
