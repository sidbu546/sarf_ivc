# SARF Circuit Classification: End-to-End Guide

This repository contains two main notebooks that together cover the full project pipeline:

- `final_update3.ipynb` (main training + evaluation + analysis notebook)
- `sarf_full_notebook_attention_rollout_consistent_v3 (2).ipynb` (training baseline variant + structural attention rollout analysis)

> Note: You referred to `sarf_full_notebook_rollout_consistent`; in this workspace the matching file is `sarf_full_notebook_attention_rollout_consistent_v3 (2).ipynb`.

---

## 1) Project Goal

The project studies **SARF fusion methods** for circuit image classification, where:

- RGB features provide semantic appearance cues.
- Edge features provide structural/topological cues.
- Fusion methods combine both to improve classification and structural sensitivity.

Core question:
- Does fusion improve model behavior, especially around **line intersections** and **connectivity structures**?

---

## 2) Notebook Responsibilities

## `final_update3.ipynb` (Primary pipeline)

This notebook is the complete end-to-end pipeline used for report-ready outputs.

### What it does
1. Loads train/validation/test splits and image paths.
2. Extracts frozen CLIP features for RGB and edge views.
3. Trains/evaluates model families:
   - Zero-shot CLIP
   - RGB linear probe
   - Edge-only probe
   - SARF-Sum
   - SARF-Concat
   - SARF-Learned Fusion
4. Runs advanced interpretability and topology analysis:
   - Topology-group performance
   - Fixed-vs-hurt error analysis vs RGB baseline
   - Class-wise gains and confusion patterns
5. Exports report-ready plots.
6. Generates Grad-CAM comparisons for RGB vs SARF-Concat.

### Main outputs produced
- Summary metrics for all methods
- Per-model prediction files
- Topology/interpretability tables
- Plot images used in the report
- Grad-CAM figure panels

Primary output folder:
- `final_sarf_paper_outputs/`

Key subfolders:
- `final_sarf_paper_outputs/plots/`
- `final_sarf_paper_outputs/advanced_interpretability/`
- `final_sarf_paper_outputs/gradcam_rgb_vs_sarf_concat/`

---

## `sarf_full_notebook_attention_rollout_consistent_v3 (2).ipynb` (Rollout-focused)

This notebook focuses on **structural attention rollout** and SAR/JAR gain analysis.

### What it does
1. Builds SARF-style baselines on frozen CLIP features.
2. Reloads CLIP vision backbone for attention extraction.
3. Computes attention rollout maps on RGB and edge streams.
4. Quantifies structure-localized attention with:
   - **SAR**: Structure Activation Ratio
   - **JAR**: Junction Activation Ratio
5. Compares rollout statistics across RGB-only, edge-only, and SARF variants.
6. Produces tables and visuals (including LaTeX-ready table text).

### Why this notebook matters
- It provides the strongest direct evidence for whether fusion increases activation over structural regions (connectivity/intersections).
- It supports claims about structural sensitivity even when top-1 improvements are modest.

---

## 3) End-to-End Workflow (Recommended Order)

1. Run `final_update3.ipynb` to:
   - generate all baseline metrics
   - produce plots and Grad-CAM artifacts
   - export report-facing performance analyses
2. Run `sarf_full_notebook_attention_rollout_consistent_v3 (2).ipynb` to:
   - compute rollout maps
   - produce SAR/JAR gains and significance-oriented structural analysis
3. Use generated artifacts to build the final report/paper.

---

## 4) Model Variants Used

- **RGB-only**: semantic-only probe.
- **Edge-only**: structure-only probe.
- **SARF-Sum**: element-wise fusion of RGB and edge embeddings.
- **SARF-Concat**: concatenation-based fusion.
- **SARF-Learned Fusion (MLP)**: learnable nonlinear fusion.

---

## 5) Interpretation Assets Used in the Report

From project outputs:

- Global metric plots (Top-1 / Macro-F1)
- Topology group plots
- Fixed-vs-hurt correction plot
- Grad-CAM comparison panels
- Attention rollout SAR/JAR table and rollout figures
- Methods diagram and accuracy summary image

---

## 6) Environment and Runtime Notes

- Notebooks are GPU-oriented (CLIP feature extraction and rollout are significantly faster on CUDA).
- Paths in notebook cells may reference cluster-style directories (`/projectnb/...`).
- If running locally, update base/output paths in the first setup cells before full execution.

---

## 7) Final Deliverables in This Workspace

- Paper source: `sarf_acl_report.tex`
- Compiled paper: `sarf_acl_report.pdf`
- Main generated artifacts: `final_sarf_paper_outputs/`

---

## 8) Quick Troubleshooting

- **Missing file/path errors**: verify dataset/output base paths in the first setup cells.
- **Slow runs**: confirm GPU is available and selected.
- **Empty plots/tables**: ensure earlier training/evaluation cells were executed successfully before analysis cells.
- **Inconsistent rollout outputs**: rerun rollout cells in one pass after model reload to avoid stale cache/state.

---

If you want, this README can also be split into:
- a short user-facing README, and
- a separate `docs/technical_pipeline.md` with deeper implementation detail.
