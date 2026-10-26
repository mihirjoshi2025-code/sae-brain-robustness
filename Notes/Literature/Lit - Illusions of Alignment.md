---
tags: [reading, controls, critique]
source: "Illusions of Alignment Between LLMs and Brains (follow-up critique)"
---
# Lit — Illusions of Alignment

> [!summary] One line
> A critique arguing that many reported **LLM–brain alignments are illusory** — statistical artifacts of method and evaluation rather than genuine shared computation.

## The core critique
- Brain scores are easy to **inflate**: generous splits, no proper null, small datasets, flexible preprocessing.
- Apparent "alignment" can appear even for models/features that shouldn't align, once you allow enough researcher degrees of freedom.
- Significance claimed without rigorous nulls ([[Permutation Test|permutation tests]]) or [[Noise Ceiling|noise ceilings]] is unreliable.

## How it maps to our project
- Reinforces the [[Controls and Baselines|control suite]] and **[[Contiguous Train-Test Split|contiguous splits]]** (no leakage across autocorrelated TRs).
- Motivates the **no-peeking rules**: fix [[Layer|layers]] and the [[Feature Taxonomy|taxonomy]] before seeing results.
- A guardrail for interpreting our own grid: treat small, non-replicating effects with suspicion.

## Owner
Primary reading for the **Controls & statistics** workstream. Pair with [[Lit - What Are LLMs Mapping To (Feghhi, Hadidi)]].
