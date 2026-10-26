---
tags: [reading, claims-1-4]
source: "Lepori, Kay, Tuckute, Interpreting Brain Responses to Language with Sparse Features from Language Models"
---
# Lit — Sparse Features for Brain Responses (Lepori, Kay, Tuckute)

> [!summary] One line
> You can use **sparse [[Sparse Autoencoder (SAE)|SAE]] features** (not just raw activations) to interpret brain responses to language, and some brain regions are explained by simple quantities like [[Surprisal|surprisal]]. **Source of our claims 1 & 4.**

## What they did
- Extracted [[Dense Hidden States|dense]] LLM activations and their SAE decompositions.
- Built [[Encoding Model|encoding models]] to predict fMRI from each.
- Compared sparse-feature encoding to dense-feature encoding.
- Isolated specific drivers (e.g., surprisal) for specific regions.

## What they found
- **Finding (→ Claim 1):** sparse SAE features predict brain activity **about as well as** dense hidden states — so you can switch to interpretable features without losing much.
- **Finding (→ Claim 4):** **frontal language regions** are largely explained by **surprisal alone** (next-word predictability), separable from meaning content.

## How it maps to our project
- **Claim 1** is foundational: if SAE ≈ dense fails across the grid, the interpretability story collapses. We benchmark SAE vs dense per configuration, above [[Variance-Matched Random Features|random-feature]] and [[Untrained-Model Baseline|untrained]] baselines.
- **Claim 4** gives the [[Surprisal]] baseline a double role: it's both a **control** (Claims 2–3 must beat surprisal) and a **claim** (does surprisal explain frontal ROIs across datasets?).

## What to watch
- Claim 1 may hold only at some [[Dictionary Size|SAE sizes]] or [[Layer|layers]] — that's exactly what the grid exposes.
- Surprisal estimates differ by model; our surprisal baseline must be computed consistently across configurations.
