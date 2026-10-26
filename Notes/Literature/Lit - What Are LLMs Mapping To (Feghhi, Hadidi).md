---
tags: [reading, controls]
source: "Feghhi, Hadidi et al., What Are Large Language Models Mapping to in the Brain?"
---
# Lit — What Are LLMs Mapping To (Feghhi, Hadidi)

> [!summary] One line
> High brain scores can come from **mundane or low-level properties**, not deep language understanding — which is **why our controls matter.**

## The worry it raises
When an LLM "predicts the brain well," the prediction may be riding on confounds:
- low-level timing ([[Word Rate|word rate]]),
- position in the passage ([[Sentence Position|sentence position]]),
- simple predictability ([[Surprisal|surprisal]]),
- or sheer dimensionality/variance of the feature set.

If you don't subtract these out, you credit the model for structure that any reasonable signal would capture.

## How it maps to our project
This paper is the justification for the **[[Controls and Baselines|control suite]]**:
- always include word-rate, sentence-position, and surprisal baselines,
- use [[Variance-Matched Random Features|variance-matched random features]] so SAE features must beat a same-scale random set,
- report [[Noise Ceiling|noise ceilings]] so a number is judged against what's achievable.

## Takeaway for us
A claim only "survives" if it beats these baselines **across the [[Experimental Grid|grid]]**. This paper is the reason robustness (not another positive result) is the whole point.
