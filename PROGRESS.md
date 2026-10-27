# Progress: Plain-English Overview

_Last updated: Oct 27, 2026. For the detailed technical log, see `BUILD_LOG.md`. For what comes next, see `NEXT_STEPS.md`._

## The project in one paragraph

AI language models (like GPT-2) and the human brain both process language. A popular line of research takes a model's internal activity and uses it to **predict brain activity** recorded with fMRI while people listen to stories. Recent papers go further and use **sparse autoencoders (SAEs)**, a tool that breaks the model's activity into thousands of human-readable "features" (e.g., a feature for "food words" or "past tense"). Those papers make big claims, like "semantic features explain ~94% of how well the model predicts the brain." **My project asks whether those claims hold up**, or whether they depend on which model, which SAE, and which evaluation method you happened to pick. I re-run the claims across a grid of setups with strict controls and report which ones survive.

## The 30-second version (for recruiter calls)

> "I'm running a reproducibility study at the intersection of AI interpretability and neuroscience. Recent papers claim that interpretable features inside language models line up with how the brain processes language. I built a pipeline that predicts real fMRI brain recordings from language-model activity, reproduced the standard published baseline on my laptop, and I'm now stress-testing four of those claims across different models, feature dictionaries, and datasets, with controls like random-feature and untrained-model baselines, to find out which results are real and which come from the analysis choices."

Good follow-ups to have ready:
- **Why it matters:** many brain-AI papers test a single setup. If a result disappears when you swap the model or the evaluation method, it's probably not telling us much about the brain.
- **What's technically interesting:** aligning word-by-word model activity to slow, delayed brain signals; regression with tens of thousands of outputs (one per brain location); building controls that catch false positives.
- **Constraint I'm working within:** everything runs on a MacBook Air (M4, 24 GB RAM), with no GPU cluster. That shaped which models and methods I chose.

## What's done so far

| # | What | Why it matters | Status |
|---|---|---|---|
| 1 | **Planned the study.** Read the key papers, wrote plain-English notes and a vocabulary glossary, and built a 4-phase plan with go/no-go checkpoints. | Locks the method in before seeing results, so I can't (even by accident) tune the analysis toward a desired answer. | Done |
| 2 | **Set up the code environment** on my laptop (Python, PyTorch on the Mac's GPU, regression library). | Everything else runs on this. | Done |
| 3 | **Built the prediction pipeline** and tested it on fake data where I knew the right answer. It found the planted signal (score 0.985) and correctly found nothing in pure noise (0.002). | Shows the code works before trusting it on real brain data. | Done |
| 4 | **Downloaded real brain data**: one person's fMRI recordings (11 podcast stories, ~4 GB) plus word timings for all 84 stories. | This is what we're trying to predict. | Done |
| 5 | **Reproduced the standard result ("Gate A").** Using GPT-2's internal activity, the pipeline predicts this person's brain activity on a story it never trained on. | This is the first checkpoint. Every later experiment builds on this baseline. | **Done ✅** |
| 6 | **Fixed the "best possible score" (noise ceiling).** The person heard the test story 10 times, and we predict the average of those 10 scans. An average is much less noisy than a single listen, so the best achievable score is higher than I was assuming. I applied the standard statistical correction for averaging. | The old version made scores look ~3× better than perfect, which is impossible. Now "35% of the best possible" means what it says. | Done |
| 7 | **Every run is now saved to a results table** (`results/runs.csv`): settings, random seed, code version, and scores. | Nothing lives only in terminal output, so every number in the final write-up can be traced to a run. | Done |
| 8 | **Chose and locked the GPT-2 layers before touching SAEs.** Tested all 12 layers using only the training stories (the test story was never used to choose). Best = layer 7, plus layer 3 (early) and layer 11 (late). Written to `configs/layers.yaml`; the script refuses to re-pick. | The study's no-peeking rule: if I picked layers after seeing SAE results, I could unconsciously pick the ones that favor a claim. | Done |
| 9 | **Locked the three models** by a rule written before looking: each needs public SAEs at two sizes in the early, middle and late parts of the network and must fit on the laptop. GPT-2 small (32K + 128K features), Gemma 2 2B (16K + 65K), Gemma 3 1B (16K + 262K). | The model list can't be tuned toward a result. Caveat: models 2 and 3 are both Google Gemma models; no other family passed the rule. | Done (Gemma access confirmed Oct 26) |
| 10 | **Built the controls** that make this a robustness study: an untrained (random-weight) GPT-2, scrambled "look-alike" features, a random dictionary, simple baselines (how fast words are coming, where we are in the sentence, and "surprisal" = how unexpected each word is), and a significance test. | If a fake or trivial feature set scores as well as the real one, the "brain alignment" claim is empty. | Done |
| 11 | **Found and fixed a subtle layer bug.** The library hands back GPT-2's last layer *after* a final normalization step, which is not what the public SAEs were trained on. Now every layer is read directly from the network. Only layer 11 changed; the layer choice still stands. | Feeding an SAE the wrong input would quietly ruin every SAE result for that layer. | Done |
| 12 | **Built a memory-light way to fit the big SAE feature sets.** Instead of working with 128,000 features directly, it works with a much smaller table of how similar each brain scan is to every other one. I checked that it gives the same answer as the original method (they agree to 6 parts in 10 million). | A 128K-feature dictionary is too big to fit the normal way on a laptop. Being provably identical keeps the pipeline fixed across all 12 setups. | Done |
| 13 | **Chose and locked the Gemma layers** (overnight, Oct 27), same no-peeking method, using the 25 training stories. **Gemma 2:** best = layer 13, early = 6, late = 25. **Gemma 3:** best = layer 10, early = 6, late = 25. Gemma 3's layers 10 and 11 were nearly tied (0.13764 vs 0.13756); the rule set in advance picks the higher one, so layer 10. | All three models now have locked layers, so SAE work on Gemma can start without any risk of peeking. All three show the same shape: prediction peaks in the middle of the network and falls off towards the end. | Done |

### What the Gate A numbers mean

- The model was trained on 10 stories and tested on a different one (*"Where There's Smoke"*).
- **Average score = 0.22** across the 10,000 most reliable brain locations, and **0.53** in the best 1%. The score is the correlation between predicted and actual brain activity: 0 means no relationship, 1 means perfect.
- **Best possible score (noise ceiling) ≈ 0.65**, so the model reaches **about 35% of what's achievable** for this data.
- **By layer (locked layers, test story scored only after locking):** early layer 3 = 0.20, **best layer 7 = 0.22**, late layer 11 = 0.20. Middle layers predict the brain best, the usual pattern in this literature.
- That's in line with published results for GPT-2 on this dataset, so the pipeline behaves like the ones in the literature.
- The whole run takes **about 31 seconds on my laptop**.

### What the controls show (GPT-2, best layer 7)

| Feature set | Score | % of ceiling | Brain locations significant |
|---|---|---|---|
| Real GPT-2 activity | 0.225 | 35% | 79% |
| Untrained GPT-2 (random weights) | 0.093 | 14% | 23% |
| Scrambled look-alike features | 0.059 | 9% | 13% |
| Surprisal only | 0.077 | 12% | (layer-independent) |
| Word rate only | 0.069 | 11% | (layer-independent) |

A trained model clearly beats every control, so the baseline is real signal, not an artifact of the method. Even an untrained network gets about 40% of the way, mostly by tracking how many words are being heard. That's why every claim gets compared against these controls.

## Problems I hit and solved (good interview stories)

- **A library removed the feature I planned to use.** TransformerLens 4.0 dropped its main model class, so I switched to Hugging Face `transformers`, which has a more stable API, and pulled the same internal activity directly.
- **A new Python version gave wrong math results.** NumPy miscomputes on Python 3.14, so I pinned the project to 3.12.
- **The word-timing files came in two different formats.** I wrote a parser that handles both.
- **A hidden mismatch in the model's last layer.** The library quietly applies a final normalization step to the last layer it hands back, so the SAE for that layer would have been fed the wrong input. I caught it by comparing the library's output against the network's raw output layer by layer, and fixed it before any result depended on it.
- **Aligning words to brain scans.** People say several words per second, but the scanner takes one snapshot every 2 seconds, and the brain's blood-flow response lags by several seconds. The pipeline resamples word-level features onto the scan timeline and adds time-lagged copies to model the delay.

## Claim 1 result for GPT-2 (first model done)

**The question:** do the SAE's "readable features" predict the brain about as well as the model's raw activity? Pass mark, set in advance: at least 90% as well.

**Answer for GPT-2: no, at every layer and both SAE sizes.** How badly it fails depends heavily on one analysis choice:

| How SAE features are prepared | SAE score vs raw activity (best layer) |
|---|---|
| Method fixed in advance (every feature rescaled to the same size) | about **0%**: no signal at all |
| Drop features that almost never switch on | about **77%** |
| Keep natural feature sizes (one shared scale) | about **72%** |

**What's going on, in plain terms:** most SAE features switch on for only a handful of words in the whole dataset. Rescaling every feature to the same size turns those rare features into huge spikes that drown out everything else. With either sensible fix, SAE features recover most of the signal, but they still fall short of the 90% bar.

**A second wrinkle:** a *random* dictionary (same size and sparsity, but no learned features) sometimes does as well as the real SAE. Whether the real SAE wins also depends on that one preparation choice.

**Why this matters for the project:** the original paper's claim depends on analysis choices that papers rarely report. That's exactly the kind of fragility this study is designed to find. The first result was the method fixed in advance, and it stays the headline. The other two are reported openly as a check on how sensitive the result is, not swapped in quietly.

## In progress (Oct 27)

- **Next up: Gemma features.** Save Gemma 2 and Gemma 3 activity at their locked layers, then run Claim 1 and the controls on them, the same way as for GPT-2.
- **Second dataset (Pereira 2018, people reading single sentences): paused, about 80% done.** Data for 8 people is downloaded (3.2 GB), along with each person's own language-region maps, which Claims 3–4 will need. The raw-activity baseline is finished for experiment 2 (all 8 people). Experiment 3 is done for 1 person at 2 of 3 layers; the other 5 people are left. Resuming it takes about 25 minutes and needs no new downloads. Caveat: this dataset gives only one averaged scan per sentence, so its "best possible score" can only be estimated roughly from the other people.
- **More training data: done.** 15 more podcast stories for the main person (10 → 25). Two more people's recordings (11 stories each) also came down before I narrowed the plan; kept for later.
- **Feature categories:** a first draft of the category rules (meaning, grammar, word form, prediction, other) is written and waiting for my review. It must be locked before any feature gets labeled.

## Waiting on me

- **Decide whether "how SAE features are prepared" becomes an official part of the grid** (run all 12 setups each way), or stays a one-off check.
- **Review and lock the feature-category rules** (`taxonomy/scheme.yaml`).
- **Find a second person to hand-label 200 features**, so I can measure whether two people agree on the categories.
- **Resume the paused Pereira runs** (about 25 minutes).
