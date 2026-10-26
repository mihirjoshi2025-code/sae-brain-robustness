---
tags: [design, core]
---
# Experimental Grid

**3 models × 2 SAE sizes × 2 datasets = 12 configurations.** The [[Encoding Model|encoding pipeline]] is held **fixed** across all 12 so that *only* the three axes vary. Plus a layer axis fixed per model.

## The axes
| Axis | Options | Notes |
|---|---|---|
| **Model** | 3 open LLMs with public SAEs. Candidates: GPT-2 family, Gemma 2 (+[[Gemma Scope]]), Llama 3.1 (+[[Llama Scope]]). | Final pick depends on which SAEs cover the layers we need. |
| **SAE dictionary** | 2 [[Dictionary Size|sizes]] per model (e.g., 16K and 65K features). | Tests whether conclusions depend on dictionary choice. |
| **fMRI dataset** | Naturalistic stories ([[LeBel-Huth fMRI|LeBel/Huth podcast fMRI]]) + isolated sentences ([[Pereira 2018]]). | Long-context vs short-context stimuli. |
| **Layer** | Best-predicting [[Layer|layer]] per model, plus one early and one late. | **Fixed per model before looking at SAE results** (no peeking). |

## Why these axes
- **Model:** if a claim only holds for GPT-2 but not Llama, it's a model artifact, not brain science.
- **SAE size:** [[Sparse Autoencoder (SAE)|SAE]] features can split/merge with dictionary size; a claim that depends on 16K-vs-65K is fragile.
- **Dataset:** naturalistic stories are rich but confounded; isolated sentences are clean but [[Noise Ceiling|noisy]] (few stimuli). A real effect should show in both.
- **Layer:** middle layers predict the brain best, so layer choice can dominate — we fix it up front to avoid cherry-picking.

## What stays fixed (the controlled pipeline)
- fMRI loading & alignment, [[Hemodynamic Delay|hemodynamic-delay]] handling.
- [[Ridge Regression|Ridge regression]] encoding (via [[himalaya]]).
- [[Contiguous Train-Test Split|Contiguous train/test splits]].
- Scoring + [[Noise Ceiling|noise ceilings]].
- The full [[Controls and Baselines|control suite]].

## Caching discipline
Store activations as **one file per `model × layer × dataset × SAE size`** so nobody recomputes. The grid is small (12), but activation extraction is the expensive step.

## Config files
Each of the 12 cells is a file in `configs/`. A config fully specifies: model + revision, SAE repo + size, dataset, layers, seed, split boundaries, and baseline set. Reproducibility = re-run from the config.
