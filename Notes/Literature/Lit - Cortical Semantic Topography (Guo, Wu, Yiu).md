---
tags: [reading, claims-2-3]
source: "Guo, Wu, Yiu (2026), Sparse Autoencoders Map Brain–LLM Alignment onto Cortical Semantic Topography"
---
# Lit — Cortical Semantic Topography (Guo, Wu, Yiu)

> [!summary] One line
> It's **meaning** that makes LLMs look brain-like, and different *types* of meaning map onto the brain regions neuroscience already expected. **Source of our claims 2 & 3.**

## Background you need
- LLM internals can **predict** brain activity (an [[Encoding Model|encoding model]]) — a known result.
- The **middle layers** predict best, but nobody had a clean explanation.
- Inside a layer everything is tangled ([[Superposition|superposition]]); a [[Sparse Autoencoder (SAE)|SAE]] un-blends it into thousands of readable [[Feature|features]].

## What they did
1. Ran stories through two models (GPT-2 XL, Llama 3.1), saved activations.
2. Un-blended into ~16K–32K SAE features per layer.
3. Sorted features into 5 buckets with GPT-4, human-checked a sample: [[Semantic Features|Semantic]], [[Syntactic Features|Syntactic]], [[Lexical Features|Lexical]], Prediction, Other.
4. Tested each bucket against fMRI from 8 people (~6h stories each).

## What they found
- **Finding 1 (→ Claim 2):** semantic features give ~**94%** of the full encoding performance; syntax helps a little, word-form ~0.
- **Finding 2:** explains the middle-layer mystery — bottom = word-form, middle = meaning, top = next-word prediction.
- **Finding 3 (→ Claim 3):** 5 meaning types mapped to pre-registered [[Cortical Regions|regions]]; **4 of 5** landed where predicted (objects→posterior temporal/angular gyrus, emotions→anterior temporal, social→frontal, space→angular gyrus; **events → nothing clear**).
- Also: SAE features beat simple concreteness/emotion word ratings; ablating meaning features hurts far more than ablating random ones; predicts reading speed; partial replication in Chinese/French.

## How it maps to our project
- **Claim 2** = Finding 1 (the 94%). We re-test across the grid; it hinges on the [[Feature Taxonomy|taxonomy]], so we fix categories first and report [[Inter-Rater Agreement (Kappa)|κ]].
- **Claim 3** = Finding 3 (category→region). The most exciting, least-proven claim.

## What's shaky (our read)
- Only **8 subjects**; region effects are small (~0.07–0.14).
- The **"events" category flopped** and is barely discussed.
- Used a **generic brain atlas** instead of each subject's own language areas → weakens region-vs-region claims.
- GPT-4 labeled the features — [[LM-Assisted Labeling|an AI labeling an AI]]; shared blind spots may slip past the human spot-check.
- SAE features aren't perfectly stable; change settings and they split/merge. **← exactly why we vary [[Dictionary Size|dictionary size]].**
