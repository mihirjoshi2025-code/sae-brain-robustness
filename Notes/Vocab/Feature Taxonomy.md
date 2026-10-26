---
tags: [vocab, taxonomy]
aliases: [feature taxonomy, category scheme, taxonomy, feature categories]
---
# Feature Taxonomy

**In one line:** the scheme for sorting [[Sparse Autoencoder (SAE)|SAE]] [[Feature|features]] into categories — [[Semantic Features|semantic]], [[Syntactic Features|syntactic]], [[Lexical Features|lexical]], prediction, other.

**Why it's critical:** Claim 2 ("semantic features do 94%") **depends entirely** on how features are labeled. Different schemes → different numbers.

**Our rules:** **fix the category scheme before labeling**; use [[LM-Assisted Labeling|LLM-assisted labeling]] + human validation; report [[Inter-Rater Agreement (Kappa)|κ]] (Gate B needs κ ≥ 0.7). Also match features **across dictionaries**.

**Related:** [[Semantic Features]] · [[Inter-Rater Agreement (Kappa)]] · [[LM-Assisted Labeling]]
