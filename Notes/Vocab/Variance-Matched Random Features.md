---
tags: [vocab, control]
aliases: [variance-matched random features, random features, random-feature baseline]
---
# Variance-Matched Random Features

**In one line:** a baseline set of **random features scaled to the same variance** as the real [[Sparse Autoencoder (SAE)|SAE]] features, so any score advantage must come from **content**, not from dimensionality or scale.

**Why it's a control:** random projections of model activations can predict the brain surprisingly well. Matching variance removes that confound — real SAE features must beat the random set.

**Related:** [[Untrained-Model Baseline]] · [[Controls and Baselines]] · [[Feature]]
