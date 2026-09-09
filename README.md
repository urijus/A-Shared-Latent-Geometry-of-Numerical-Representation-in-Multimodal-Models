# A Shared Latent Geometry of Numerical Representation in Multimodal Models

<p align="center">
  <b>Do multimodal models represent the same numerical result using the same internal geometry?</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key Results</a> •
  <a href="#repository-map">Repository Map</a> •
  <a href="#installation">Installation</a> •
  <a href="#reproducing-the-experiments">Reproduction</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  📄 Paper coming soon &nbsp;&nbsp;•&nbsp;&nbsp; 🧾 arXiv coming soon
</p>

---

<!--
Add a teaser figure here when available.

<p align="center">
  <img src="assets/overview.png" width="850" alt="Overview of the shared latent numerical geometry">
</p>

<p align="center">
  <em>
    Numerical representations occupy condition-specific coordinates,
    while preserving a common relational geometry across operations and modalities.
  </em>
</p>

---
-->

## Overview

Do models represent the same number similarly when it is obtained through different arithmetic operations or from text rather than an image?

This repository contains the research code accompanying **A Shared Latent Geometry of Numerical Representation in Multimodal Models**.

We study numerical representations in the multimodal `Gemma4-12B-it` model using controlled arithmetic tasks. Matched addition and subtraction problems are presented either as **text** or as **rendered images**, allowing us to vary the computational route and input modality while holding the target numerical result fixed.

Rather than treating representational similarity as a single notion, the experiments distinguish progressively stronger questions:

1. **Linear accessibility** — can numerical variables be decoded from the residual stream?
2. **Causal relevance** — can intervening on a low-dimensional numerical subspace change the generated result?
3. **Cross-condition causal reuse** — do causal directions learned in one operation or modality remain effective in another?
4. **Shared latent geometry** — does numerical structure remain after removing directions coupled to the current digit readout?
5. **Autoregressive reuse** — is that latent numerical identity subsequently transformed into the readout needed for later generated digits?

The central result is that **shared numerical representation does not require identical neural coordinates**.

Addition and subtraction exhibit substantial direct causal reuse within a modality. Changing modality creates a larger mismatch, but a learned orthogonal transformation and global scale recover most of the causal effect of a subspace learned directly in the target condition.

After separating directions associated with the digit the model is about to emit, the remaining readout-orthogonal component still organizes numerical identity similarly across text/image and addition/subtraction. These condition-specific geometries can be synchronized into a common relational coordinate frame.

Finally, interventions show that this latent numerical identity is functionally used during generation: modifying it redirects subsequent digits while largely preserving the immediate digit, and its effect becomes aligned with digit-readout directions at the next autoregressive step.

---

## Paper

**A Shared Latent Geometry of Numerical Representation in Multimodal Models**
**Oriol Juan Sabater**

- **Paper:** coming soon
- **arXiv:** coming soon
- **Code:** this repository
- **Thesis:** Master's Thesis in Computer Vision, September 2026

The public manuscript link and final citation metadata will be added once the paper is released.

---

## Key Results

### 1. Arithmetic results develop structured causal representations

Numerical result information becomes strongly organized in the middle-to-late layers of the model.

Linear and Fourier probes reveal structured numerical geometry, while activation patching and **Distributed Alignment Search (DAS)** identify corresponding low-dimensional residual-stream subspaces whose interventions can change the generated arithmetic result.

Addition and subtraction converge onto related late causal states. Multiplication follows a more dynamic trajectory involving intermediate place-value representations.

→ [`experiments/01_arithmetic_reference/`](experiments/01_arithmetic_reference/)

---

### 2. Operations share causal directions more directly than modalities

The four core conditions are

| Symbol | Operation | Input modality |
| --- | --- | --- |
| `T+` | Addition | Text |
| `T-` | Subtraction | Text |
| `I+` | Addition | Image |
| `I-` | Subtraction | Image |

Directly reusing a DAS space across **addition and subtraction within the same modality** retains substantial causal effectiveness.

Changing modality produces a larger mismatch: text-trained and image-trained causal spaces are related, but do not occupy identical residual-stream coordinates.

→ [`experiments/02_cross_condition_transfer/`](experiments/02_cross_condition_transfer/)

---

### 3. Cross-modal numerical changes are alignable

For a source condition \(i\) and destination condition \(j\), numerical displacements are related using a scaled orthogonal map

\[
A_{j \leftarrow i} = \alpha Q,
\qquad
Q^\top Q = I.
\]

The orthogonal component \(Q\) changes orientation, while the scalar \(\alpha\) changes the global magnitude of the displacement.

For modality-changing relations, combining the learned orientation and scale recovers approximately **93% of the destination-trained causal effect** on average.

An unrestricted linear transformation provides only a small additional gain, indicating that most of the useful cross-modal correspondence is captured by a rotation/reflection together with one global scale.

The fitted transformation describes a relationship between representations; it is **not** claimed to be a transformation explicitly computed by the model during its forward pass.

---

### 4. Numerical geometry survives outside the current digit readout

The final pre-generation DAS space has dimension \(k=22\).

We rotate this space according to its coupling with the centered digit-unembedding span and obtain the exact decomposition

\[
R = C \oplus L,
\]

where

- \(C\) is the **9-dimensional readout-coupled component**, and
- \(L\) is the **13-dimensional readout-orthogonal component**.

By construction,

\[
U_{\mathrm{digit}}^\top L \simeq 0.
\]

Although \(L\) alone has little immediate autoregressive effect, it remains strongly organized by numerical identity.

Across all directed relations between `T+`, `T-`, `I+`, and `I-`, the exact \(L\) spaces achieve approximately

- **0.828** mean held-out transition cosine;
- **75.3%** top-1 retrieval among numerical values excluded from alignment fitting;
- **0.742** mean no-fit cross-condition RSA.

Shuffling numerical identities collapses the effect to chance.

→ [`experiments/03_readout_latent_geometry/`](experiments/03_readout_latent_geometry/)

---

### 5. The latent spaces admit a common numerical frame

Pairwise transformations between the four condition-specific \(L\) spaces can be synchronized into one orientation \(U_i\) and scale \(s_i\) per condition.

A numerical displacement in condition \(i\) can therefore be expressed in common coordinates as

\[
\Delta u
=
\frac{1}{s_i} U_i^\top \Delta z_i^L,
\]

and reconstructed in condition \(j\) through

\[
\Delta z_j^L
=
s_j U_j \Delta u.
\]

Replacing independently fitted pairwise maps with synchronized maps preserves almost all of the held-out numerical geometry:

| Mapping | Transition cosine | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| Independent pairwise | 0.831 | 76.3% | 98.9% |
| Full synchronization | 0.821 | 76.7% | 99.3% |
| Leave-one-relation-out | 0.795 | 72.1% | 98.8% |

In the leave-one-relation-out test, the relation between two conditions is removed entirely during synchronization and reconstructed from the remaining condition geometry.

The four spaces are therefore not merely alignable two at a time: their numerical relationships are mutually compatible with a **single latent relational frame**.

This common frame is a coordinate description of the geometry, not a privileged residual-stream subspace shared literally by all conditions.

→ [`experiments/04_global_geometry/`](experiments/04_global_geometry/)

---

### 6. Modality and operation structure predicts a held-out condition

The controlled \(2\times2\) design allows one complete modality–operation condition to be held out and predicted from the remaining three.

The missing-corner geometry reaches approximately

- **0.696** held-out transition cosine;
- **62.1%** top-1 retrieval among unseen numerical values.

A shuffled-correspondence control returns to chance.

This suggests that modality and operation effects are partially reusable across condition-specific latent geometries rather than being arbitrary pairwise relations.

---

### 7. Latent numerical identity is reused during autoregressive generation

The causal role of \(C\) and \(L\) differs across autoregressive time.

At the final pre-generation position:

- the immediate digit predominantly follows the value encoded by \(C\);
- changing numerical identity only inside \(L\) redirects the later continuation.

For two-digit values that share their first digit but differ in their second, replacing only the \(L\) component redirects generation toward the alternative numerical value while leaving the shared first digit essentially unchanged.

At the next autoregressive position, after propagating the intervention through the preserved KV cache, the induced change becomes aligned with the digit-readout geometry corresponding to the new continuation.

The conversion becomes strong immediately after the first layer able to access intervention-affected cached states.

Thus, numerical identity can be stored in a representation with no direct coupling to the **current** digit readout and later become output-facing when the corresponding digit is required.

→ [`experiments/05_autoregressive_transfer/`](experiments/05_autoregressive_transfer/)

---

## Experimental Setting

The primary configuration uses the instruction-tuned multimodal

```text
Gemma4-12B-it
```

model.

The main cross-condition experiments compare matched addition and subtraction problems presented as either text or rendered images.

Text and image versions contain identical

- operands,
- arithmetic operation,
- target result, and
- auxiliary numerical labels.

In the visual condition, the arithmetic expression appears only in the image. The textual message specifies only the required answer format.

The main causal reference configuration uses

```text
Layer:           43
Position:        final pre-generation position
Residual site:   resid_post
DAS dimension:   k = 22
DAS seeds:       3
```

Layer 43 is used as a common reference point in the late region where numerical representations are stable and causal interventions are strong. It is not intended as a claim that every condition reaches its absolute causal maximum at exactly this layer.

<details>
<summary><b>Why the final pre-generation position?</b></summary>

<br>

For textual inputs, earlier positions correspond to operands, operators, and chat-template elements.

These positions do not have direct semantic counterparts in the visual condition.

The final pre-generation position provides a natural common site across modalities:

- the complete input has already been processed;
- text and image prompts share the same final channel token;
- several transformer blocks remain before the final output readout.

</details>

---

## Repository Map

The public experiment entrypoints follow the organization of the paper.

| Paper section | Scientific question | Entrypoints | Implementation |
| --- | --- | --- | --- |
| **V. Arithmetic reference map** | Where and in what dimension is numerical result information represented? | [`experiments/01_arithmetic_reference/`](experiments/01_arithmetic_reference/) | [`src/experiments/arithmetic_reference/`](src/experiments/arithmetic_reference/) |
| **VI. Cross-condition transfer** | Are operations and modalities represented in shared coordinates, or only through alignable geometry? | [`experiments/02_cross_condition_transfer/`](experiments/02_cross_condition_transfer/) | [`src/experiments/cross_condition_transfer/`](src/experiments/cross_condition_transfer/) |
| **VII A–C. Readout and latent geometry** | What numerical structure remains after separating the current digit readout? | [`experiments/03_readout_latent_geometry/`](experiments/03_readout_latent_geometry/) | [`src/experiments/readout_latent_geometry/`](src/experiments/readout_latent_geometry/) |
| **VII D–F. Global geometry** | Can condition-specific latent spaces be described in one globally consistent frame? | [`experiments/04_global_geometry/`](experiments/04_global_geometry/) | [`src/experiments/global_geometry/`](src/experiments/global_geometry/) |
| **VIII. Autoregressive transfer** | Is latent numerical identity causally reused during later generation? | [`experiments/05_autoregressive_transfer/`](experiments/05_autoregressive_transfer/) | [`src/experiments/autoregressive_transfer/`](src/experiments/autoregressive_transfer/) |

Supplementary robustness analyses and cross-model checks live under

[`src/experiments/supplementary/`](src/experiments/supplementary/).

Figure-generation entrypoints follow the same paper order under

[`visualizations/`](visualizations/).

---

## Repository Structure

```text
configs/                  lightweight, versioned experiment configuration

experiments/              paper-ordered command-line entrypoints

src/
  common/                 deterministic seeding and structured I/O
  data/                   arithmetic and multimodal dataset generation
  experiments/            scientific implementations grouped by paper claim
  geometry/               alignment, subspace, readout, and synchronization math
  interventions/          DAS and causal intervention machinery
  models/                 model aliases, loading, and residual-stream access
  probes/                 linear and Fourier probe utilities
  representations/        activation extraction and representation helpers
  training/               supplementary model-training utilities

visualizations/           paper and supplementary figure generation

scripts/                  repository validation and utilities
```

Large generated artifacts are intentionally excluded from version control.

<details>
<summary><b>Local experimental workspace</b></summary>

<br>

A working copy may additionally contain

```text
.local/banks/             activation and probe banks
.local/cluster_scripts/   archived machine-specific launchers
dataset/                  generated arithmetic and image datasets
models/                   downloaded model weights
results/                  experiment outputs and regenerated figures
```

These directories are not part of the source release.

</details>

---

## Installation

Clone the repository and create an isolated Python environment.

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv
```

Activate the environment.

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Install the runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for the accelerator and CUDA environment of the target machine if it is not already provided by the environment.

Model aliases and Hugging Face identifiers are defined in

[`src/models/hf.py`](src/models/hf.py).

When a matching local directory exists under `models/`, the loader uses it. Otherwise, it falls back to the configured Hugging Face identifier.

Access-controlled models require the corresponding Hugging Face permissions, credentials, and license acceptance.

---

## Reproducing the Experiments

Experiment scripts under `experiments/` are intentionally thin entrypoints. Scientific implementations live under `src/experiments/`.

The wrappers do not change the experiment defaults or output schemas.

Every major entrypoint exposes its configuration through

```bash
--help
```

### 1. Arithmetic reference map

```bash
python experiments/01_arithmetic_reference/linear_probes.py --help

python experiments/01_arithmetic_reference/fourier_probes.py --help

python experiments/01_arithmetic_reference/das.py --help
```

These experiments cover

- linear accessibility;
- Fourier numerical geometry;
- activation patching;
- DAS localization;
- arithmetic-operation comparison;
- multiplication intermediate variables;
- Fourier/DAS relationships.

---

### 2. Cross-condition causal transfer

```bash
python experiments/02_cross_condition_transfer/causal_transfer.py --help

python experiments/02_cross_condition_transfer/align_subspaces.py --help
```

These experiments cover

- direct DAS reuse;
- destination-normalized causal transfer;
- scaled orthogonal alignment;
- transformation controls;
- rank-restricted transport;
- cross-operation transport;
- composed modality/operation transformations.

---

### 3. Readout and latent geometry

```bash
python experiments/03_readout_latent_geometry/readout_audit.py --help
```

These experiments cover

- digit-unembedding geometry;
- readout ablations;
- repeat-number controls;
- exact \(C/L\) decomposition;
- held-out numerical geometry;
- random and shuffled controls;
- representational similarity analysis.

---

### 4. Global numerical geometry

```bash
python experiments/04_global_geometry/synchronize_frames.py --help
```

These experiments cover

- pairwise \(L\)-space alignment;
- orthogonal group synchronization;
- synchronized condition frames;
- leave-one-relation-out reconstruction;
- modality-only factorization controls;
- missing-corner prediction.

---

### 5. Autoregressive latent transfer

```bash
python experiments/05_autoregressive_transfer/latent_identity_swap.py --help
```

These experiments cover

- matched and mismatched \(C/L\) interventions;
- continuation redirection;
- repeat-number latent geometry;
- KV-cache propagation;
- latent-to-readout conversion across generation steps.

---

## Figures

Plotting code mirrors the organization of the experiments.

For example:

```bash
python visualizations/02_cross_condition_transfer/plot_direct_transfer.py --help

python visualizations/04_global_geometry/plot_common_frame.py --help

python visualizations/05_autoregressive_transfer/plot_next_step_readout.py --help
```

Generated figures are not committed to the source repository.

To reproduce a manuscript figure, first run the corresponding experiment and generate its empirical summary files, then invoke the matching plotting entrypoint.

---

## Generated Data and Results

The Git repository contains

- source code;
- lightweight configuration;
- validation scripts;
- documentation.

It intentionally excludes large or generated research artifacts, including

- downloaded model weights;
- generated datasets;
- activation banks;
- learned DAS checkpoints;
- probe checkpoints;
- NumPy arrays;
- JSON / JSONL experiment summaries;
- generated tables;
- figures;
- audio and video files;
- site-specific cluster launchers.

A clean clone therefore contains **no empirical result archive**.

Reported figures must be regenerated from the corresponding experiment outputs.

Some historical result directories retain names such as

```text
final_exps
closing
```

These paths are deliberately unchanged because archived experiments and downstream analyses depend on them.

---

## Validation

Fast repository-level validation can be run with

```bash
python -m compileall -q src experiments visualizations scripts

python scripts/validate_repository.py
```

These checks catch

- Python syntax errors;
- broken imports;
- packaging regressions;
- missing paper-facing command-line entrypoints.

They are **not** substitutes for scientific reproduction.

Full reproduction requires rerunning the relevant GPU experiments with the documented

- model checkpoint;
- datasets;
- behavioral filtering;
- random seeds;
- layers;
- token positions;
- DAS dimensions;
- intervention pairs;
- optimization settings;
- evaluation protocols.

---

## Notes on Interpretation

### Shared geometry does not mean identical neural coordinates

The main claim is relational.

The condition-specific numerical representations do not generally occupy one identical set of residual-stream directions. Instead, corresponding numerical relationships can remain compatible under simple transformations between their coordinate systems.

---

### Learned transformations are analysis tools

The fitted transformations

\[
A_{j \leftarrow i} = \alpha Q
\]

describe correspondences between learned representations.

They should not be interpreted as evidence that the model explicitly computes \(Q\) or \(\alpha\) during its forward pass.

---

### The common frame is not a privileged residual-stream subspace

Synchronization identifies one orientation and scale per condition whose pairwise relations explain the observed numerical geometry.

The resulting common coordinates are defined only up to a global orthogonal transformation and scale.

The common frame should therefore be interpreted as a **relational coordinate description**, not as a unique physical subspace of the model.

---

### \(C\) and \(L\) are an analytical decomposition

The decomposition

\[
R = C \oplus L
\]

is constructed inside the learned DAS space according to coupling with the current digit-discriminative readout.

`C` and `L` should not be interpreted as evidence that the model explicitly implements two separate neural modules.

---

## Citation

Citation metadata will be updated when the public manuscript is released.

For now:

```bibtex
@misc{juan2026sharedlatentgeometry,
  title  = {A Shared Latent Geometry of Numerical Representation in Multimodal Models},
  author = {Juan Sabater, Oriol},
  year   = {2026},
  note   = {Master's Thesis in Computer Vision. Manuscript forthcoming.}
}
```

Once the arXiv version is available, please cite the public manuscript instead.

---

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
