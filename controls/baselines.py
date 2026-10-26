"""Simple low-level baselines: word rate, sentence position, surprisal.

Each returns word-level features (N_words, d) that go through the same Lanczos ->
FIR -> ridge path as every other feature set.

- word_rate: one constant feature per word; after Lanczos it is words per TR.
- sentence_position: [index of the word in its sentence, 1/(index+1)]. The
  ds003020 TextGrids have no punctuation and no transcripts are stored locally,
  so sentences are approximated by pauses (NEXT_STEPS D13): a new sentence starts
  when the gap between one word's offset and the next word's onset is >= 0.5 s.
  This gives ~14-17 words per "sentence" on UTS01 stories; it is an
  approximation (pauses also fall mid-sentence and some boundaries have none).
- surprisal: -log p(word | left context) in nats, summed over the word's
  subtokens, from the trained model with the same tokenization (" " + word) and
  1024/512 sliding windows as the activations; a BOS token is prepended so the
  first word is scored too. Cached as `{model}_{dataset}_surprisal.h5`.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import torch

from features.cache import ACT_ROOT
from features.extract import get_device
from pipeline import data as D

PAUSE_S = 0.5


def word_rate(story: str) -> np.ndarray:
    return np.ones((len(D.parse_textgrid_words(story)), 1), dtype=np.float32)


def sentence_position(story: str, pause: float = PAUSE_S) -> np.ndarray:
    w = D.parse_textgrid_words(story)
    on = np.array([a for _, a, _ in w])
    off = np.array([b for _, _, b in w])
    start = np.r_[True, (on[1:] - off[:-1]) >= pause]
    sent = np.cumsum(start) - 1
    idx = np.arange(len(w)) - np.flatnonzero(start)[sent]
    return np.stack([idx, 1.0 / (idx + 1)], 1).astype(np.float32)


@torch.no_grad()
def word_surprisal(bundle, words: list[str], max_len: int = 1024, stride: int = 512,
                   device: str | None = None) -> np.ndarray:
    """words (N,) -> (N,) surprisal in nats (sum over subtokens)."""
    model, tok = bundle
    device = device or get_device()
    ids, owner = [tok.bos_token_id], [-1]
    for wi, w in enumerate(words):
        t = tok(" " + w, add_special_tokens=False)["input_ids"] or \
            tok(w, add_special_tokens=False)["input_ids"]
        ids += t
        owner += [wi] * len(t)
    ids, owner = np.array(ids), np.array(owner)
    n = len(ids)
    lp = np.full(n, np.nan)
    for s in range(0, n, stride):
        e = min(s + max_len, n)
        logits = model(input_ids=torch.tensor(ids[s:e][None], device=device)).logits[0]
        logp = torch.log_softmax(logits.float(), -1)[:-1]  # predicts tokens s+1..e-1
        tgt = torch.tensor(ids[s + 1:e], device=device)[:, None]
        got = logp.gather(1, tgt)[:, 0].cpu().numpy()
        g = np.arange(s + 1, e)
        keep = g >= (s + 1 if s == 0 else s + stride)  # most-left-context rule
        lp[g[keep]] = got[keep]
        if e == n:
            break
    assert not np.isnan(lp[1:]).any()
    surp = np.zeros(len(words))
    np.add.at(surp, owner[1:], -lp[1:])
    return surp


def surprisal_path(model: str, dataset: str = "lebel") -> str:
    return f"{ACT_ROOT}/{model.replace('/', '-')}_{dataset}_surprisal.h5"


def surprisal(story: str, model: str = "gpt2", dataset: str = "lebel",
              _bundle=[]) -> np.ndarray:
    """(N_words, 1) cached surprisal; loads the causal LM only on a cache miss."""
    path = surprisal_path(model, dataset)
    if os.path.exists(path):
        with h5py.File(path, "r") as h:
            if story in h:
                return h[story][:][:, None]
    if not _bundle or _bundle[0][0] != model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"loading {model} LM head for surprisal...")
        lm = AutoModelForCausalLM.from_pretrained(model).to(get_device()).eval()
        _bundle[:] = [(model, (lm, AutoTokenizer.from_pretrained(model)))]
    s = word_surprisal(_bundle[0][1], [w for w, _, _ in D.parse_textgrid_words(story)])
    os.makedirs(ACT_ROOT, exist_ok=True)
    with h5py.File(path, "a") as h:
        h.create_dataset(story, data=s.astype(np.float32))
    return s.astype(np.float32)[:, None]
