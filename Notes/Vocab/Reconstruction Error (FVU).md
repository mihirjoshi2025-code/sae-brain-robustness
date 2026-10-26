---
tags: [vocab, sae]
aliases: [FVU, fraction of variance unexplained, reconstruction error]
---
# Reconstruction Error (FVU)

**In one line:** how much of the model's original activity an [[Sparse Autoencoder (SAE)|SAE]] **fails** to rebuild from its features. 0 = perfect, 1 = useless.

**Why we care:** if an SAE throws away a lot of the signal, SAE features could predict the brain worse simply because information was lost, not because "sparse" is worse. We log FVU next to every SAE score so [[Claims Under Test|Claim 1]] can be read fairly.

**Example:** GPT-2 layer 7, 32K OpenAI SAE on our stories: FVU ≈ 0.26 (about three quarters of the activity is captured).

**Related:** [[Sparse Autoencoder (SAE)]] · [[Dictionary Size]] · [[Dense Hidden States]]
