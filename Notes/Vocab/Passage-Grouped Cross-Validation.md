---
tags: [vocab, control]
aliases: [passage-grouped CV, leave-passages-out, grouped cross-validation]
---
# Passage-Grouped Cross-Validation

**In one line:** when splitting [[Pereira 2018]] sentences into train and test, keep **every sentence from a passage on the same side** of the split.

**Why it's a control:** sentences in a passage share a topic and words, so a model could "predict" a test sentence by remembering its neighbors from the same passage. Holding out whole passages is the sentence-level version of a [[Contiguous Train-Test Split|contiguous split]]. The ridge penalty is also chosen by an inner passage-grouped split inside the training passages only, so the held-out passages are never used for model selection.

**Our setup:** 8 outer folds (passage *p* goes to fold *p* mod 8, which spreads each topic's passages across folds), 5 inner folds.

**Related:** [[Contiguous Train-Test Split]] · [[Pereira 2018]] · [[Encoding Model]]
