---
tags: [vocab]
aliases: [layer, layers, model layer]
---
# Layer

**In one line:** one step in the LLM's processing stack (like a floor in a building); text enters at the bottom, a prediction leaves the top.

**Key fact:** the **middle layers** predict the brain best — bottom layers track word-form, middle track meaning, top track next-word prediction.

**Our rule:** pick the best-predicting layer per model, plus one early and one late, and **fix them before looking at SAE results** so we don't cherry-pick.

**Related:** [[Dense Hidden States]] · [[Experimental Grid]] · [[Semantic Features]]
