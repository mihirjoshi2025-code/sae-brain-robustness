---
tags: [vocab]
aliases: [dictionary size, SAE size, dictionary, number of features]
---
# Dictionary Size

**In one line:** how many [[Feature|features]] an [[Sparse Autoencoder (SAE)|SAE]] has (e.g., 16K vs 65K) — the "vocabulary" of concepts it can split activations into.

**Why we care:** it's one of the three grid axes. Bigger dictionaries split concepts more finely; a claim that only holds at one size is fragile. Testing 2 sizes per model checks whether conclusions depend on dictionary choice.

**Related:** [[Sparse Autoencoder (SAE)]] · [[Experimental Grid]] · [[Feature]]
