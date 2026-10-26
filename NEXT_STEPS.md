# NEXT_STEPS — agent execution spec

_Last updated: 2026-10-25. Audience: an AI coding agent working solo in this repo. Human-readable summary of finished work: `PROGRESS.md`. Detailed history: `BUILD_LOG.md`. Protected rules: `CLAUDE.md` (read it first; it wins any conflict with this file)._

## 0. How to use this file

- Work tasks **in ID order** unless `Depends` allows otherwise. One task per session is the normal unit.
- Before starting a task: read its block, the modules it names, and §2 (invariants). Do not re-derive decisions recorded in §4.
- A task is done only when **every Acceptance line passes**. Then: set its row in §1 to `done`, commit code + configs (never data), and append a 3–6 line entry to `BUILD_LOG.md` (what was built, key numbers, surprises).
- If a task hits a decision marked `ASK` in §4, stop and ask the user; do not pick silently. Anything not marked `ASK` uses the stated default.
- If reality contradicts this file (an API changed, a dataset has a different shape), fix the code to match reality, then correct this file in the same commit.
- Prefer the smallest run that answers the question: top-10k voxels, 1 subject, 1 layer first. Scale up only after a small run passes.

## 1. Status board

| ID | Task | Status | Depends |
|---|---|---|---|
| T0 | Spearman-Brown ceiling, `results/runs.csv` logger, word-level activation cache | done | — |
| T1 | Freeze GPT-2 layers (`scripts/sweep_layers.py` → `configs/layers.yaml`) | done (best 7, early 3, late 11) | T0 |
| T2 | SAE availability check → lock model list (`configs/models.yaml`) | done except HF login + Gemma license (**ASK**) | T1 |
| T3 | Hook alignment + SAE feature extraction (GPT-2) | done | T2 |
| T4 | High-dimensional encoding path (precomputed linear kernel ridge) | done (parity PASS) | T3 |
| T5 | Claim 1 first pass: GPT-2 dense vs SAE (2 sizes), podcast data | done: **fails** at all 3 layers × 2 sizes under the pre-registered pipeline; SAE < random dictionary flagged; normalization **ASK** (D15) | T4 |
| T6 | Control suite (untrained, random dictionary, word rate, sentence position, surprisal, block permutation) | done (random dictionary runs inside `run_claim1.py`) | T4 |
| T7 | Data expansion: more podcast stories; Pereira 2018 loader + baseline | todo | T0 (can run in parallel with T3–T6) |
| T8 | Config-driven grid runner over all cells | todo | T5, T6, T7 |
| T9 | Models 2–3: layer sweep, extraction, grid | todo | T8 |
| **Gate B** | Controls on every cell; κ ≥ 0.7 | — | T9, T10 |
| T10 | Feature taxonomy: scheme, labels, κ, cross-dictionary matching | todo | T3 (scheme part can start any time) |
| T11 | Score claims 2–4 | todo | T9, T10 |
| **Gate C** | Every claim scored with ceilings + p-values | — | T11 |
| T12 | Figures, results grid, write-up, public README | todo | Gate C |

## 2. Invariants (never violate)

**Method (from CLAUDE.md, restated as checks):**
1. Layers are frozen in `configs/layers.yaml` and **committed before any SAE feature is encoded or scored** for that model. `sweep_layers.py` refuses to overwrite an existing entry; never edit that file by hand.
2. The taxonomy category scheme (`taxonomy/scheme.yaml`) is committed **before** any labeling run.
3. Splits are contiguous at story level (podcast) or passage level (Pereira). Never shuffle TRs or sentences across folds. FIR delays are applied per story so lags never cross story boundaries (`pipeline/lebel.fit_score` already does this; keep that property in new paths).
4. Model selection (layers, alphas, thresholds) never touches the test story `wheretheressmoke` or a held-out Pereira passage.
5. Every scored run goes through `pipeline/results.log_run`. Every random step takes an explicit seed (default `0`) and logs it.
6. Every reported score is shown next to its noise ceiling, and every claim verdict has a p-value or CI.
7. Missing a gate means cutting scope (§5 cut order), never skipping controls.

**Data / storage:**
- Data root: `~/brain-llm-data/` (outside the repo). fMRI: `~/brain-llm-data/ds003020/`. Activations: `~/brain-llm-data/activations/`. Put new datasets under `~/brain-llm-data/<dataset>/`.
- Never commit `.hf5/.h5/.mat/.nii`, model weights, or activation files (`.gitignore` covers most of them; add patterns for new formats).
- Small derived artifacts that are fine to commit: `configs/*.yaml`, `results/runs.csv`, `taxonomy/labels/*.jsonl`, `results/figures/*.png|svg`.
- Cache rule: one activation file per `model × layer × dataset × feature-set`. Compute each once. Check the cache before running any model.

**Compute (MacBook Air M4, 24 GB unified memory, no external GPU, ~72 GB free disk):**
- Keep peak process RSS ≤ 16 GB. Before allocating any array over 2 GB, compute its size and pick a chunked/sparse path instead.
- Disk budget for this project: ≤ 40 GB total under `~/brain-llm-data/`.
- Models run on `mps` (`features.extract.get_device`). Weights in bf16 for models over 1B params; cast activations to float32 before caching. Ridge runs on CPU (himalaya `numpy` backend) unless a benchmark shows that `torch_mps` is faster and numerically equal (max |Δcorr| < 1e-3).
- Any single job expected to run longer than 30 min: run it in the background, log to a file, and make it resumable (skip already-cached stories/cells).
- Python env: `.venv` (Python 3.12, pinned; 3.14 breaks numpy). Run with `.venv/bin/python`. Add dependencies to `pyproject.toml`.

## 3. Current codebase (what exists; reuse, don't rewrite)

| Module | Key API | Notes |
|---|---|---|
| `pipeline/data.py` | `load_bold(story, subject)`, `load_repeats(story, subject)` → `(n_rep,T,V)`, `parse_textgrid_words(story)` → `[(word,on,off)]`, `tr_times(n_bold)`, `trim_to_bold(stim, n_bold)` | TR=2.0 s; stim grid = n_bold+10, trim 5 each end; V=81126 for UTS01. |
| `pipeline/downsample.py` | `lanczos_matrix(word_times, tr_grid)`, `lanczos_downsample(word_feats, word_times, tr_grid)` | Built as a `(n_TR × n_words)` matrix; sparse input stays sparse. Identical to the old per-TR loop (max diff 0.0). |
| `pipeline/encoding.py` | `add_fir_delays(X, delays)`, `zscore_fit`, `contiguous_split`, `noise_ceiling_from_repeats`, `spearman_brown`, `noise_ceiling_corr(repeats)` → `{ceiling,…}` | `ceiling` = sqrt(Spearman-Brown reliability of the 10-repeat mean). Normalized score = corr / ceiling. |
| `pipeline/lebel.py` | `TRAIN_STORIES`, `TEST_STORY`, `ALPHAS`, `story_folds(lengths)`, `tr_features(wfeats, story, n_bold)`, `fit_score(...)`, `fit_score_kernel(...)`, `summarize(scores, ceiling)` | The single scoring path. Both solvers take `cv="story"` (default: whole-story alpha folds), `info=` (best alphas, alive columns) and `return_pred=`. `fit_score_kernel` = same ridge in the dual with a precomputed linear kernel from sparse matrices; z-scoring is done algebraically (`A=X/sd`, mean correction), so wide SAEs are never densified. This replaces the planned `pipeline/kernel_encoding.py`; kernels are not cached on disk (fits take ~2 min). |
| `pipeline/results.py` | `log_run(row)`, `COLUMNS` | Unknown keys raise. Adding to `COLUMNS` auto-migrates the existing CSV header on the next write (`_migrate_header`). Columns include `control, sae_release, p_value, p_value_median, n_perm, frac_voxels_sig, n_sig_voxels_fdr, n_features_used, layer_used, ci_low, ci_high, cv`. |
| `features/extract.py` | `load_model(name)` → `(model, tok)`, `get_device()` | HF `transformers`. TransformerLens is not used. |
| `features/word_features.py` | `word_activations(bundle, words, layer)` | Int or list of layers. Last-subtoken activation per word; 1024-token windows, stride 512. **Layer L = raw output of decoder block L via forward hook** (`blocks.L.hook_resid_post`), not `hidden_states[L+1]` (the last one is post-final-norm). L0–L10 bit-identical to the old cache; L11 rebuilt. Works for `.h` (GPT-2) and `.layers` (Gemma/Llama/Qwen). |
| `features/cache.py` | `cache_path(model, layer, dataset, feats)`, `word_feats(bundle_fn, model, story, layers, dataset, feats="dense")` | Files: `{model}_L{layer}_{dataset}_{feats}.h5`, one dataset per story. GPT-2 L0–L11 dense cached for the 11 UTS01 stories; untrained at L3/7/11. |
| `features/sae.py` | `sae_spec(model, size, layer)`, `load_sae`, `encode_sparse`, `sae_word_feats(model, layer, size, story)`, `sae_fvu(...)` | Replaces the planned `features/sae_features.py`. CSR codes cached as `..._sae{width//1024}k.h5` (group per story with FVU attr). Firing counts are derived from the CSR on demand (`getnnz`). |
| `controls/` | `untrained.py`, `random_features.py` (column shuffle), `random_dictionary.py` (T6 spec), `baselines.py`, `stats.py` (`block_permutation`, `fdr_bh`, `block_bootstrap_ci`, `autocorr_length`) | `scripts/run_controls.py --frozen` runs untrained / random_matched / word_rate / sentence_position / surprisal with permutation tests. |
| `scripts/run_gate_a.py` | Gate A run (`--layer N` or `--frozen`) | L9: corr 0.223, ceiling 0.645, normalized 0.346. |
| `scripts/sweep_layers.py` | Per-model layer sweep + freeze | Refuses to overwrite a frozen entry. |
| `scripts/check_saes.py` | T2 registry scan + model-3 rule | Widths parsed from ids; verify with `SAE.from_pretrained`. |
| `scripts/test_kernel_parity.py` | T4 parity | Passes if same-alpha voxels agree < 1e-3 and ≤ 0.1% of voxels flip alpha on near-tied CV scores (L9: 6e-7, 1/10000 flip). |
| `scripts/run_claim1.py` | T5 | dense + 2 SAE sizes + random dictionary per size, permutation p, bootstrap ratio CI, verdict. |
| `scripts/smoke_test.py` | Synthetic sanity test | Must keep passing. |

Empty dir reserved for new code: `taxonomy/`.

## 4. Decisions

| # | Decision | Value | Status |
|---|---|---|---|
| D1 | Model 1 | `gpt2` (small, 12 layers, d=768) | fixed |
| D2 | Model 2 | `google/gemma-2-2b` (26 layers, d=2304, ~5 GB bf16) with Gemma Scope residual SAEs | default; confirm in T2 |
| D3 | Model 3 | `google/gemma-3-1b-pt` (Gemma Scope 2 res 16k + 262k, all 26 layers): the only model passing the T2 rule. Same family as model 2 (no non-Gemma model passes). | fixed by rule (T2) |
| D4 | SAE sizes | Two widths per model, ideally ~16K and ~65K (or nearest available, ratio ≥ 4×) | T2 |
| D5 | Subjects | UTS01 first. UTS02 and UTS03 only if T9 finishes with time left. | default |
| D6 | Podcast training stories | Grow from 10 to ~25 for UTS01 (T7). Test story stays `wheretheressmoke`. | default |
| D7 | Voxels | Top-10k by single-repeat reliability (fixed from the test repeats, same set for every cell). Full cortex only for final figures. | default |
| D8 | Pereira sentence feature | Last-token activation of the sentence, run with preceding sentences of the same passage as context | default |
| D9 | Feature labels | Neuronpedia explanations where available; otherwise our own top-activating-context auto-interp | default |
| D10 | Mapping labels → categories | LLM classification with a fixed prompt. Needs an API key and has a small cost. | **ASK** before T10 runs it |
| D11 | Second human rater for κ | — | **ASK** (needs a person) |
| D12 | Cortical ROIs for claims 3–4 | Check whether ds003020 ships pycortex subject DBs / ROI masks. Pereira ships language-network parcels. | resolve in T11; **ASK** if neither works |
| D13 | Sentence boundaries in podcast data | TextGrids lack punctuation. Default: boundary = pause ≥ 0.5 s between words. | default (document as approximation) |
| D15 | Sparse-feature normalization for the official Claim 1 / grid | Pre-registered: z-score every alive column (T4). Under it SAE ≈ 0 because rare columns dominate. Post-hoc diagnostic (`scripts/diag_sae_normalization.py`, L7): frequency filter → SAE .173 vs dense .225 (77%); global scale → .149 vs .205 (72%); SAE-vs-random-dictionary ordering flips between the two. Claim 1 fails under all three, by very different margins. Report the pre-registered result as primary either way; whether to add a sensitivity axis to the grid is the user's call. | **ASK** |
| D14 | Claim thresholds | Use the source paper's operationalization (see `Notes/Literature/`). Otherwise use the defaults in T11. | default |

## 5. Scope cut order (when a gate slips)
1. Llama 3.1 8B / any stretch goal → 2. extra subjects → 3. Pereira dataset → 4. model 3.
Minimum viable result: 2 models × 2 SAE sizes × podcast data, full controls.

---

## 6. Task specs

### T1 — Freeze GPT-2 layers *(done)*
- If `scripts/sweep_layers.py --model gpt2` is still running, wait for it; do not start a second sweep.
- **Acceptance:** `configs/layers.yaml` has a `gpt2` entry `{best, early, late, date}` · 60 `layer_selection` rows (12 layers × 5 folds) in `runs.csv` · `configs/layers.yaml` committed before any T3 code is run.

### T2 — SAE availability → lock models
- **Build** `scripts/check_saes.py`: enumerate SAELens's pretrained registry (`sae_lens` `pretrained_saes.yaml` / `get_pretrained_saes_directory()`; the import path varies by version, so check the installed one). For each candidate model, print the release, hook point, width, and layer coverage.
- **Candidates to verify** (from memory, unconfirmed): GPT-2 small → OpenAI `resid-post-v5-32k` / `-128k`, `gpt2-small-res-jb` (24576, single width). Gemma-2-2b → `gemma-scope-2b-pt-res(-canonical)` 16k/65k. Small third models → Pythia-70m/160m SAEs (e.g., SAEBench releases with multiple widths). Stretch → `llama_scope_lxr_8x/32x`.
- **Model-3 rule:** must have ≥ 2 widths at a residual hook in the early, middle and late thirds of the network · ≤ 8 GB bf16 · a different family from GPT-2/Gemma preferred · tie-break by Neuronpedia label coverage.
- **SAE-per-layer rule:** for each frozen layer, use the residual SAE whose hook matches that layer (see T3). If no width pair exists at that exact layer, use the nearest available layer, record it as `layer_used` with a reason, and do **not** re-pick layers.
- **Output:** `configs/models.yaml`: `{model_id, hf_name, n_layers, d_model, dtype, saes: [{width, release, sae_id_template, hook}]}` per model.
- **Acceptance:** 3 models listed, each with 2 widths resolvable via `SAE.from_pretrained` at all of its frozen layers (Gemma and model 3 get their layers frozen in T9; check coverage across the whole network now) · Gemma license accepted on HF (else **ASK** the user to accept it).

### T3 — Hook alignment + SAE feature extraction
- **Hook mapping (critical):** HF `hidden_states[0]` = embeddings = `resid_pre 0`; `hidden_states[i]` = `resid_post (i−1)` = `resid_pre i`. **Exception:** the last element of `hidden_states` has the final norm applied (GPT-2 `ln_f`, Gemma final norm), so it is not the raw residual. For the last layer, capture the block output with a forward hook instead. The existing `gpt2_L11` cache is post-`ln_f`: rebuild it as `feats="dense_raw"` or via hook before using it for SAEs, and record which one the dense baseline used.
- Write `features/hooks.py: resid(model_id, layer, hook) -> extraction spec` and make `word_activations` use it. Do not change the cached L0–L10 semantics.
- **Model-specific:** Gemma 2 → load with `attn_implementation="eager"` (softcapping) and prepend BOS to every window. Drop the BOS position from outputs for any model where BOS is prepended. Follow each SAE's `prepend_bos`/context-size config where present.
- **Build** `features/sae_features.py`:
  - `load_sae(release, sae_id)`: handle both SAELens return styles (tuple vs object).
  - `encode_story(sae, dense_words) -> scipy.sparse.csr_matrix (N_words, width)`: use `sae.encode` so the SAE's own input normalization is applied; batch 512 words; float32.
  - Cache: `{model}_L{layer}_{dataset}_sae{width}.h5`; per story store `data, indices, indptr, shape`.
  - Also store per-feature firing counts per story (needed for dead-feature filtering and taxonomy).
- **Validation (per model × layer × width), log to `runs.csv` with `purpose=sae_check`:** fraction of variance unexplained (FVU) of `sae.decode(encode(x))` vs `x` on 2,000 words, and mean L0.
- **Acceptance:** FVU < 0.4 for every SAE used (if any fails, hook alignment is wrong: fix it, don't relax the threshold) · L0 within 2× of the SAE's reported L0 when one is published · GPT-2 SAE caches exist for both widths at the 3 frozen layers for all UTS01 stories · disk use reported.

### T4 — High-dimensional encoding path
- **Problem:** 65K features × 4 delays × ~9k TRs × float32 ≈ 9 GB. Dense ridge is infeasible, so use **precomputed linear kernels** (mathematically identical to ridge with the same alpha).
- **Build** `pipeline/kernel_encoding.py`:
  1. `tr_sparse(sparse_words, story, n_bold)`: build the Lanczos matrix `L` once per story (`n_TR × n_words`), then `L @ sparse` (it stays manageable: n_TR rows × active columns).
  2. Feature filter: keep features with nonzero activation in ≥ 1 **training** story. The filter is computed on training stories only and logged as `n_features_used`.
  3. Z-score each column using **training** statistics, computed in column chunks (≤ 4,096 columns per chunk).
  4. Kernels: `K_tr = Σ_delays Σ_chunks X_d X_dᵀ` (`n_tr × n_tr`) and `K_te = Σ X_te,d X_tr,dᵀ`, with delays applied per story. Cache kernels to `~/brain-llm-data/cache/kernels/{cell}_{split}.npy`.
  5. Fit `himalaya.kernel_ridge.KernelRidgeCV(kernel="precomputed", alphas=ALPHAS, cv=<story-level folds as explicit (train_idx, val_idx) list>)`. Z-score Y on train; score with per-voxel correlation.
- **Parity test** (`scripts/test_kernel_parity.py`): GPT-2 L9 dense through the kernel path vs `pipeline.lebel.fit_score`. Note that `RidgeCV` uses its own internal CV, so pass the same explicit story folds to both, or fix the alpha for the comparison.
- **Acceptance:** parity max |Δcorr| < 1e-3 across voxels that select the same alpha, with ≤ 0.1% of voxels flipping alpha on near-tied CV scores (exact equality is impossible at a tie) · peak RSS for a 65K-wide GPT-2 cell ≤ 16 GB (measure with `resource.getrusage`) · wall time per cell logged.

### T5 — Claim 1 first pass (GPT-2, podcast)
- **Script:** `scripts/run_claim1.py --model gpt2`. For each frozen layer, score dense, SAE-small and SAE-large on the test story with the top-10k voxel set.
- **Paired comparison:** block bootstrap (blocks of 10 TRs, 1,000 resamples, seed 0) on per-voxel `corr_SAE − corr_dense`, averaged over voxels → 95% CI. Also report `ratio = normalized_mean_SAE / normalized_mean_dense`.
- **Pre-registered verdict:** *holds* if the lower 95% CI bound of the ratio is ≥ 0.9; *fails* if the upper bound is < 0.9; otherwise *inconclusive*.
- **Acceptance:** 9 rows (3 layers × 3 feature sets) with ceilings · verdict per layer · a short entry in `BUILD_LOG.md`. Report the number; no plots needed yet.

### T6 — Control suite (`controls/`)
Every control is a feature set that plugs into the same encoding path and logs with `features=<control_name>`.
- `controls/untrained.py`: `AutoModelForCausalLM.from_config(cfg)` with `torch.manual_seed(0)`. Cache as model `{model}-untrained`. Dense only, at the same frozen layers.
- `controls/random_dictionary.py` (variance-matched random features): a fixed Gaussian random projection `R (d_model × width)` (seed 0) of the same dense activations, then ReLU, then keep the top-k per word where k = the real SAE's L0 for that word. Sort columns by variance and rescale each to match the real SAE's sorted per-feature variances. Same width as the SAE it controls.
- `controls/baselines.py`, each mapped to TRs by the same Lanczos path:
  - `word_rate` — 1 feature (a one-hot per word, downsampled).
  - `sentence_position` — word index within its sentence (sentence boundaries per D13), plus 1/(index+1).
  - `surprisal` — `−log p(word | context)` summed over subtokens, from the same model and windows as the activations. Cache `{model}_{dataset}_surprisal.h5`. The trained model is required, so compute it alongside extraction.
- `controls/stats.py`:
  - `block_permutation_p(y, pred, block=10, n=1000, seed=0)`: permute blocks of predictions, recompute per-voxel corr, one-sided p. Never refit.
  - `fdr_bh(p, q=0.05)`.
  - `block_bootstrap_ci(diff_fn, …)`: reused by T5 and T11.
- Extend `results.COLUMNS` (via `migrate_columns`) with `control`, `p_value_median`, `n_sig_voxels_fdr`, `n_features_used`, `layer_used`, `ci_low`, `ci_high`.
- **Acceptance:** on GPT-2 L9 the controls run end-to-end and log rows · sanity orderings hold: dense > untrained, and SAE ≥ random dictionary (if not, flag it in `BUILD_LOG.md`; do not "fix" it, because it may be a real finding) · permutation test on a shuffled-target sanity run gives ≈ 5% voxels at p < 0.05.

### T7 — Data expansion *(can run in background alongside T3–T6)*
- **Podcast:** generalize `scripts/download_data.sh` to take `SUB` and a story list from args. Add ~15 more UTS01 stories (~200 MB each). Update `pipeline/lebel.TRAIN_STORIES` to read from `configs/data.yaml` instead of the hard-coded list. **Freeze the new story list before rerunning anything.** The GPT-2 layer choice stays frozen; do not re-sweep with more data.
- **Pereira 2018:** find the data on OSF (the Pereira et al. 2018 project page; verify the URL) → `~/brain-llm-data/pereira2018/`. Experiments 2 (384 sentences) and 3 (243 sentences). Write `pipeline/pereira.py`:
  - `load(subject, experiment)` → `(Y: n_sent × V, sentences, passage_ids, topic_ids, network_masks)`.
  - Features per D8. No FIR delays (one image per sentence).
  - CV = leave-passages-out (folds by `passage_id`; never split a passage).
  - Noise ceiling: if per-presentation data is available, use split-half repeats with Spearman-Brown. Otherwise follow the Brain-Score (Schrimpf et al. 2021) subject-pool extrapolation and record which method was used.
- **Acceptance:** `configs/data.yaml` lists subjects/stories/Pereira subjects · dense GPT-2 baseline on Pereira logged with its ceiling · disk total ≤ 40 GB.

### T8 — Grid runner
- `configs/cells/{model}__{dataset}__sae{width}.yaml` (12 files), generated by `scripts/make_cells.py` from `models.yaml × data.yaml × layers.yaml`. Schema: `{cell_id, model, layers:{best,early,late}, sae:{release,sae_id,width}, dataset, subjects, voxel_selection, delays, alphas, seed}`.
- `scripts/run_cell.py <cell.yaml> [--layers best]`: dense + SAE + all T6 controls; skips any (cell, layer, feature set, control) combination that already has a row in `runs.csv` with the same `git_commit` prefix (resumable).
- **Acceptance:** `run_cell.py` on the 4 GPT-2 cells completes and is idempotent (a second invocation logs nothing new).

### T9 — Models 2 and 3
- For each model: `sweep_layers.py --model <id>` (dense only) → commit `layers.yaml` → T3 extraction + validation → `run_cell.py` for its 4 cells.
- Expected compute: Gemma-2-2b forward passes over ~25 stories take minutes, not hours. The ridge/kernel fits dominate run time.
- **Acceptance (Gate B, part 1):** all 12 cells have dense, SAE and control rows for the best layer, plus early/late layers for dense and SAE.

### T10 — Feature taxonomy (`taxonomy/`)
- `taxonomy/scheme.yaml` (commit first): categories `semantic, syntactic, lexical/orthographic, prediction/position, other`, each with a one-paragraph definition, 3 positive examples, 2 near-miss negatives, and tie-break rules.
- **Which features get labels:** features alive in our stimuli (firing count > 0 in training stories) at the best layer, for every SAE in the grid.
- **Labels (D9):** fetch Neuronpedia explanations (use their bulk exports if available; otherwise the per-feature API, rate-limited, cached to `taxonomy/labels/{model}_{sae}.jsonl`). For uncovered features, write explanations from the top-20 activating word contexts in our own stories.
- **Category assignment (D10, ASK):** `taxonomy/classify.py` uses a fixed prompt (scheme text + explanation + top contexts) → one category plus a confidence score. Cache results. Prompt, model and date go in the file header.
- **Agreement:** sample 200 features, stratified across SAEs (seed 0) → `taxonomy/rating_sheet.csv` for two human raters (D11). `taxonomy/kappa.py` reports Cohen's κ for human–human and for human–LLM. If κ < 0.7: refine the **guidelines text only** (never the categories), re-rate a fresh 200-feature sample, recompute.
- **Cross-dictionary matching:** for each model, compute the cosine similarity of decoder rows (`W_dec`) between the small and large SAE → best match per small-SAE feature. Report the fraction with cosine > 0.7 and the category agreement among matched pairs.
- **Acceptance (Gate B, part 2):** κ ≥ 0.7 (human–human) reported · every alive feature has a category · matching stats logged.

### T11 — Score claims 2–4
All analyses use the best layer; repeat at early/late layers only if time allows. Every verdict gets a CI or p-value and is shown next to the ceiling.
- **Claim 2 (semantic share ≈ 94%):** fit using only semantic-category features vs all features. `share = mean r²(semantic) / mean r²(all)` over FDR-significant voxels. **Null:** 100 random feature subsets of the same size (seed 0..99). Verdict: report share ± CI and its percentile vs the null. The claim "semantic features explain most of it" *holds* only if share ≥ 0.8 **and** share > the 95th percentile of the size-matched null.
- **Claim 3 (categories → distinct regions):** per-category models (or banded ridge with one band per category if memory allows) → per voxel, the winning category. Statistic: between-ROI vs within-ROI separation of category preference (or spatial autocorrelation if ROIs are unavailable, per D12). **Null:** shuffle category labels across features (1,000×) and recompute.
- **Claim 4 (frontal ≈ surprisal):** in frontal language ROIs (IFG / MFG language parcels), compare surprisal-only vs full-dense prediction. `ratio = r²(surprisal) / r²(dense)` with block-bootstrap CI. *Holds* if the CI lower bound is ≥ 0.8 (or the source paper's threshold, per D14).
- **Output:** `results/claims.csv` with columns `claim, cell, layer, statistic, value, ci_low, ci_high, null_pct, ceiling, verdict ∈ {holds, weakens, fails, inconclusive}`.
- **Acceptance (Gate C):** every claim × cell has a verdict row with a ceiling and a CI or p-value.

### T12 — Write-up
- `scripts/make_figures.py` → `results/figures/`: (1) claims × cells verdict heatmap; (2) per-cell SAE vs dense normalized-score bars with control bars; (3) cortical maps for claim 3 if ROIs are available; (4) κ and matching summary.
- Public `README.md`: question, grid, how to reproduce (download → extract → `run_cell.py` → figures), key results table. Keep `PROGRESS.md` in plain English, updated with final results.
- **Acceptance:** fresh clone + documented commands reproduce `results/claims.csv` for one cell end-to-end.

---

## 7. Resource reference (back-of-envelope; check before large allocations)

| Object | Size |
|---|---|
| BOLD, 1 story, all voxels | ~350 TRs × 81k × 4 B ≈ 110 MB |
| Y train, 25 stories, top-10k voxels | ~9k × 10k × 4 B ≈ 360 MB (whole cortex ≈ 2.9 GB) |
| Linear kernel, train | 9k × 9k × 4 B ≈ 325 MB |
| Dense word activations, 25 stories (Gemma) | ~50k words × 2304 × 4 B ≈ 460 MB per layer |
| SAE word features, sparse | ~50k words × L0 (~50–100) × 8 B ≈ 40 MB per layer per width |
| Delayed dense SAE matrix (avoid) | 9k × 4 × 65k × 4 B ≈ 9.4 GB |
| Gemma-2-2b bf16 weights | ~5 GB · Llama-3.1-8B bf16 ≈ 16 GB (stretch only) |
