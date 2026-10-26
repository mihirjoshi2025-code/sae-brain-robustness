"""Fetch Pereira et al. 2018 sentence-experiment data (exps 2 and 3).

Source: OSF project https://osf.io/crwz7/ (wiki -> "Datasets" table of contents at
evlab.mit.edu/sites/default/files/documents/index2.html), which links one Dropbox tar
per participant (~4 GB each, mostly exp-1 files we do not need). To stay inside the
disk budget we STREAM each tar and write only the sentence-experiment members
`data_384sentences.mat` / `data_243sentences.mat` (the `examples_*` variants hold an
identical sentence matrix but lack the language-network atlases); everything else is
read past and discarded. `pipeline/pereira.convert` then turns each .mat into a
float32 HDF5 and deletes the .mat (re-run this script to get it back). Members are extracted with tarfile's `data` filter (no links,
no absolute paths) into ~/brain-llm-data/pereira2018/raw/<SUBJ>/. Nothing from the
download is executed. Also fetches the stimulus text files and the OSF materials zip.

Run: .venv/bin/python -I scripts/download_pereira.py [SUBJ ...]   (resumable)
"""
from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

DEST = Path.home() / "brain-llm-data" / "pereira2018"
DBX = "https://www.dropbox.com/s/{}?dl=1"
# participant -> (dropbox id, experiments with sentence data); from the TOC page
SUBJECTS = {
    "P01": ("5duv9fgigrzx817/P01.tar", (2, 3)), "M02": ("n5yfb2cupd9zmwk/M02.tar", (2, 3)),
    "M03": ("4pr8slvkrpxaeek/M03.tar", (3,)), "M04": ("q8qeuf5johz0ic1/M04.tar", (2, 3)),
    "M07": ("jvtxgv4oplqgmix/M07.tar", (2, 3)), "M08": ("ch2j6feyahcc3l5/M08.tar", (2,)),
    "M09": ("l8bou2bn7qzjucy/M09.tar", (2,)), "M14": ("9chvc7rofyvb4gr/M14.tar", (2,)),
    "M15": ("xlijoxhgcm39z6v/M15.tar", (2, 3)),
}
N_SENT = {2: 384, 3: 243}
EXTRAS = {  # small text / materials files
    "stimuli_384sentences.txt": "xisek7q1hxw2p2l/stimuli_384sentences.txt",
    "stimuli_243sentences.txt": "ro7ijpiqwx2qq4a/stimuli_243sentences.txt",
    "stimuli_384sentences_dereferencedpronouns.txt":
        "k8g1b6ac1t2gg85/stimuli_384sentences_dereferencedpronouns.txt",
    "stimuli_243sentences_dereferencedpronouns.txt":
        "d5v709btx08bhme/stimuli_243sentences_dereferencedpronouns.txt",
    "DMN_overlap_n60.zip": "obbx7mgmzd8t3h2/DMN_overlap_n60.zip",
}
OSF_ZIP = ("Pereira_Materials.zip", "https://osf.io/download/hmgv2/")


def fetch(url: str, out: Path) -> None:
    if out.exists():
        return
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(out)
    print(f"  {out.name}: {out.stat().st_size / 1e6:.1f} MB", flush=True)


def fetch_subject(subj: str) -> None:
    """Stream the whole tar once; keep every sentence-experiment .mat for this subject."""
    dbx, exps = SUBJECTS[subj]
    out = DEST / "raw" / subj
    if (out / ".done").exists():
        print(f"skip {subj}")
        return
    out.mkdir(parents=True, exist_ok=True)
    keep = tuple(f"data_{n}sentences.mat" for n in (N_SENT[e] for e in exps))
    print(f"stream {subj} (keep *{keep})", flush=True)
    with urllib.request.urlopen(DBX.format(dbx)) as r, tarfile.open(fileobj=r, mode="r|") as tf:
        for m in tf:
            name = Path(m.name).name
            ok = m.isfile() and name in keep
            print(f"    member {m.name} {m.size / 1e6:.0f} MB {'KEEP' if ok else ''}", flush=True)
            if ok:
                m.name = name  # flatten; the data filter still vets it
                tf.extract(m, out, filter="data")
    got = sorted(p.name for p in out.glob("*.mat"))
    print(f"  {subj}: {got}", flush=True)
    if set(keep) <= set(got):
        (out / ".done").touch()
    else:
        print(f"  WARNING {subj}: missing some of {keep}", flush=True)


def main(subjs: list[str]) -> None:
    (DEST / "raw").mkdir(parents=True, exist_ok=True)
    (DEST / "stimuli").mkdir(exist_ok=True)
    for name, dbx in EXTRAS.items():
        fetch(DBX.format(dbx), DEST / "stimuli" / name)
    fetch(OSF_ZIP[1], DEST / "stimuli" / OSF_ZIP[0])
    for s in subjs or list(SUBJECTS):
        fetch_subject(s)


if __name__ == "__main__":
    main(sys.argv[1:])
