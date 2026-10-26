"""Load ds003020 (LeBel/Huth) preprocessed BOLD + parse TextGrid word timings.

Conventions (verified against this dataset):
  - TR = 2.0 s. Preprocessed `data` is (n_TRs, 81126 voxels).
  - Full stimulus TR grid length = n_bold + 10; align by trimming 5 TRs each end.
  - `wheretheressmoke.hf5` has `individual_repeats` (n_rep, n_TRs, V) for noise ceiling.
  - Word tier = item [2]; filter non-speech tokens ({NS}, sp, etc.).
"""
from __future__ import annotations

import os
import re
import h5py
import numpy as np

TR = 2.0
TRIM = 5  # TRs trimmed from each end of the downsampled stimulus
DATA_ROOT = os.path.expanduser("~/brain-llm-data/ds003020")
NONWORDS = {"", "sp", "{NS}", "{SL}", "{BR}", "{LG}", "{CG}", "{LS}", "{NS}", "{SP}"}


def bold_path(story: str, subject: str = "UTS01") -> str:
    return f"{DATA_ROOT}/preprocessed/{subject}/{story}.hf5"


def load_bold(story: str, subject: str = "UTS01") -> np.ndarray:
    with h5py.File(bold_path(story, subject), "r") as h:
        return h["data"][:].astype(np.float32)


def load_repeats(story: str = "wheretheressmoke", subject: str = "UTS01") -> np.ndarray:
    with h5py.File(bold_path(story, subject), "r") as h:
        return h["individual_repeats"][:].astype(np.float32)  # (n_rep, T, V)


def _clean(out):
    res = []
    for w, a, b in out:
        w = w.strip()
        if w in NONWORDS or w.startswith("{"):
            continue
        res.append((w.lower(), float(a), float(b)))
    return res


def parse_textgrid_words(story: str):
    """Return list of (word, onset, offset) from the word tier.

    Handles both Praat formats found in ds003020:
      - long format: `item [2]: ... intervals [n]: xmin/xmax/text`
      - chronological format: interleaved `<tier> xmin xmax` + `"text"` lines.
    """
    txt = open(f"{DATA_ROOT}/textgrids/{story}.TextGrid").read()

    if "chronological" in txt[:200].lower():
        lines = [ln.strip() for ln in txt.splitlines()]
        # Determine which tier number is "word" (order of IntervalTier decls).
        tier_names = re.findall(r'"IntervalTier" "(.*?)"', txt)
        word_tier = tier_names.index("word") + 1  # 1-indexed
        out = []
        hdr = re.compile(r'^(\d+)\s+([\d.]+)\s+([\d.]+)\s*$')
        for i, ln in enumerate(lines):
            m = hdr.match(ln)
            if m and int(m.group(1)) == word_tier:
                text = lines[i + 1].strip().strip('"') if i + 1 < len(lines) else ""
                out.append((text, m.group(2), m.group(3)))
        return _clean(out)

    # long format
    m = re.search(r'item \[2\]:(.*)', txt, flags=re.S)
    block = m.group(1)
    pat = re.compile(
        r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"',
        flags=re.S,
    )
    out = [(text, xmin, xmax) for xmin, xmax, text in pat.findall(block)]
    return _clean(out)


def tr_times(n_bold: int) -> np.ndarray:
    """TR center times for the FULL (untrimmed) grid: length n_bold + 2*TRIM."""
    n_full = n_bold + 2 * TRIM
    return (np.arange(n_full) + 0.5) * TR


def trim_to_bold(stim_full: np.ndarray, n_bold: int) -> np.ndarray:
    """Trim TRIM TRs from each end -> length n_bold."""
    out = stim_full[TRIM:-TRIM]
    assert out.shape[0] == n_bold, f"{out.shape[0]} != {n_bold}"
    return out
