# SARF Circuit Classification — End-to-End Project Guide

This project evaluates **frozen CLIP** representations for circuit image classification, then studies whether **RGB–edge fusion (SARF)** improves both **accuracy** and **structural sensitivity** (junctions, intersections, connectivity).

The work is split across two notebooks in a deliberate order:

1. `sarf_full_notebook_attention_rollout_consistent_v3 (2).ipynb` — **primary modeling notebook**: baselines, fusion variants, and **attention rollout** analysis.
2. `final_update3.ipynb` — **secondary analysis notebook**: **Grad-CAM**, **topology** analysis, and additional report-ready diagnostics/plots.

> Naming note: you referred to `sarf_full_notebook_rollout_consistent`; in this repository the corresponding file is `sarf_full_notebook_attention_rollout_consistent_v3 (2).ipynb`.

---

## Part A — `sarf_full_notebook_attention_rollout_consistent_v3 (2).ipynb`

This notebook is the **start-to-finish modeling and fusion experiment** notebook.

### What it establishes

- **Dataset + preprocessing**: loads circuit images, constructs paired RGB and edge views, and prepares train/validation/test splits.
- **Frozen CLIP backbone**: extracts **512-d** embeddings for RGB and edge streams using a frozen CLIP vision encoder.
- **Baselines (from scratch in this notebook’s pipeline)**:
  - Zero-shot CLIP (when included in the run configuration)
  - RGB-only linear classifier on RGB embeddings
  - Edge-only linear classifier on edge embeddings
- **Fusion variants (SARF family)**:
  - **SARF-Sum** (element-wise addition of embeddings)
  - **SARF-Concat** (concatenation + linear classifier on the fused vector)
  - **SARF-Learned Fusion (MLP)** (nonlinear fusion module)
  - **Ensemble** variants when enabled in the experiment configuration (as reflected in your results table)

### What it measures (task performance)

- Standard classification metrics such as **Top-1 accuracy**, **Macro-F1**, and **Top-5** behavior (depending on what is enabled for each method).

### What it measures (structural interpretability)

This notebook’s distinguishing contribution is **attention rollout** on CLIP ViT:

- Reloads a CLIP vision model suitable for attention extraction.
- Computes rollout heatmaps for RGB and edge pathways (and fused weightings where applicable).
- Summarizes structure-localized attention using:
  - **SAR** (Structure Activation Ratio)
  - **JAR** (Junction Activation Ratio)
- Reports **gains vs RGB-only** and associated statistical summaries used to argue that fusion increases activation around **intersections/connectivity** regions.

### Typical outputs you should expect

- Trained/evaluated model comparisons across baselines and fusion variants.
- Rollout-derived **tables/figures** and (when executed) LaTeX-friendly table text for the paper.

---

## Part B — `final_update3.ipynb`

This notebook is used **after** the main modeling + rollout study to generate **additional interpretability and stratified analyses** that are especially useful for a written report.

### What it adds beyond Part A

- **Topology-aware evaluation**:
  - groups examples by structural complexity
  - analyzes how performance and edge-signal usage behave across topology bands
- **Error behavior vs RGB baseline**:
  - “fixed RGB mistakes” vs “hurt RGB-correct” style accounting to show selective fusion effects
- **Grad-CAM comparisons** (RGB linear probe vs SARF-Concat):
  - qualitative panels for regimes like both-correct, both-wrong, and RGB-wrong/SARF-correct
- **Report-ready plots**:
  - global metric summaries
  - topology plots
  - prediction-collapse diagnostics (especially important for unstable fusion settings)

### Typical outputs you should expect

- Plot images and interpretability exports suitable for inclusion in a PDF report.
- Grad-CAM figure panels for qualitative evidence alongside rollout statistics.

---

## Recommended execution order

1. Run `sarf_full_notebook_attention_rollout_consistent_v3 (2).ipynb` end-to-end to produce:
   - baseline + fusion results
   - attention rollout + SAR/JAR evidence
2. Run `final_update3.ipynb` to produce:
   - Grad-CAM panels
   - topology and additional interpretability plots/tables

This ordering matches how the evidence is layered: **quantitative structural attention first**, then **qualitative saliency + topology stratification** for interpretation.

---

## Practical notes

- These notebooks were authored for a GPU environment; CLIP feature extraction and rollout are much slower on CPU.
- Some cells contain **absolute paths** from a shared compute environment. If you run locally, update the dataset and output directory variables at the top of each notebook before executing all cells.

---

## Report artifacts in this workspace

- LaTeX source: `sarf_acl_report.tex` (and an alternate build `sarf_acl_report_final.tex` if present)
- Compiled PDFs: `sarf_acl_report.pdf`, `sarf_acl_report_final.pdf`
- Generated figures/tables folders (as produced by the notebooks): `final_sarf_paper_outputs/` and related export directories
