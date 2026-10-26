---
tags: [vocab]
aliases: [SAE, sparse autoencoder, SAEs]
---
# Sparse Autoencoder (SAE)

**In one line:** a tool that takes the tangled internal numbers of an LLM and splits them into thousands of separate [[Feature|features]], each doing roughly one thing.

**The smoothie analogy:** the model's activation is a blended smoothie; the SAE un-blends it back into strawberry, banana, spinach. "Sparse" = only a few ingredients are active for any given word.

**Why we care:** the whole project rests on using SAE features (interpretable) instead of raw [[Dense Hidden States|dense activations]] (opaque). Claim 1 asks whether that swap loses accuracy.

**Gotcha:** SAE features aren't perfectly stable — change the [[Dictionary Size|dictionary size]] and features split or merge. That's exactly why the grid varies SAE size.

**Related:** [[Feature]] · [[Dictionary Size]] · [[Superposition]] · [[SAELens]]
