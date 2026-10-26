---
tags: [overview, week1]
---
# Project Overview

> [!summary] The whole project in 5 lines
> - Several recent papers claim that **interpretable AI features** (from sparse autoencoders) predict **brain activity** and even map onto specific **brain regions**.
> - Those results were each shown with **one model, one SAE, one dataset**. That's fragile.
> - We re-run the key claims across a **grid of 12 configurations** (3 models × 2 SAE sizes × 2 fMRI datasets), holding the pipeline fixed.
> - We add **strict controls** (random-feature baselines, untrained models, surprisal baselines, noise ceilings) so we don't fool ourselves.
> - Output: a reproducible pipeline + a results grid + a short write-up saying **which claims survive**.

## The question in plain English
When you feed a story to an AI language model and record its internal numbers, you can use those numbers to **predict** what a person's brain does while hearing the same story. Recent work used [[Sparse Autoencoder (SAE)|sparse autoencoders]] to break those numbers into readable **features** (objects, emotions, grammar…) and claimed:
- the **meaning** features do almost all the predicting,
- different **types** of meaning line up with different **brain regions**.

Exciting if true. But each result depended on specific choices of model, dictionary, and dataset. **Our core question:** do these results reflect how the brain really processes language, or are they **artifacts** of those choices?

## What makes this a *robustness* study
Instead of producing one more "look, it aligns!" result, we vary the three choices that could be driving everything and see what holds:
1. **Model** — swap GPT-2 / Gemma 2 / Llama 3.1.
2. **Dictionary** — swap SAE size (e.g., 16K vs 65K features).
3. **Dataset** — swap long naturalistic stories vs short isolated sentences.

If a claim only holds for one cell of the grid, it's probably an artifact. If it holds across the grid, it's probably real.

## Why controls are the heart of it
A high "brain score" can be fake. A random projection of model activations can predict the brain surprisingly well (see [[Lit - What Are LLMs Mapping To (Feghhi, Hadidi)]] and [[Lit - Illusions of Alignment]]). So every number we report is compared against:
- an [[Untrained-Model Baseline|untrained model]],
- [[Variance-Matched Random Features|variance-matched random SAE features]],
- simple baselines ([[Word Rate]], [[Sentence Position]], [[Surprisal]]),
- and the [[Noise Ceiling]] (the best any model could do given noisy brain data).

See [[Controls and Baselines]].

## The end products
1. A **reproducible pipeline** anyone can run from a config file.
2. A **results grid** across all 12 configurations.
3. A **short write-up** stating which published claims survive and which choices flip them.

## How we work
- One shared GitHub repo, folders per workstream (`pipeline/`, `features/`, `taxonomy/`, `controls/`) + a `configs/` folder with one file per configuration.
- 14-week semester, 4 phases, 3 gates. See [[_Roadmap]] in `Plans/`.
- The [[Roles and Workstreams|encoding pipeline]] is the critical path; the other three workstreams build against cached test data until it's ready.
