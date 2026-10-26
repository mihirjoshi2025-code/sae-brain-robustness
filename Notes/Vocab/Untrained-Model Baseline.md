---
tags: [vocab, control]
aliases: [untrained-model baseline, untrained model, random init baseline]
---
# Untrained-Model Baseline

**In one line:** run the whole encoding pipeline using a **randomly-initialized** (untrained) model, to see how much score comes from **architecture alone**, not from anything the model learned.

**Why it's a control:** a surprising amount of brain-prediction can come from a model's structure/dimensionality even with random weights. A real "the model understands meaning" effect must **beat** this.

**Related:** [[Variance-Matched Random Features]] · [[Controls and Baselines]] · [[Lit - What Are LLMs Mapping To (Feghhi, Hadidi)]]
