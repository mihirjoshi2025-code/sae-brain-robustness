#!/usr/bin/env bash
# Fetch preprocessed ds003020 (LeBel/Huth) stories for one subject.
# Usage: scripts/download_more_subjects.sh SUB [STORY ...]
#   default STORY list = the 11 used for UTS01 Gate A (10 train + wheretheressmoke).
#   e.g. scripts/download_more_subjects.sh UTS01 $(yq '.podcast.train_stories_added[]' configs/data.yaml)
# Resumable: skips files that already exist and open with h5py. Downloads go to a
# .part file first; each file is verified with h5py (read-only, `python -I`).
set -euo pipefail

S3=s3://openneuro.org/ds003020/derivatives/preprocessed_data
DEST=~/brain-llm-data/ds003020/preprocessed
PY="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"
[ $# -ge 1 ] || { echo "usage: $0 SUB [STORY ...]" >&2; exit 1; }
SUB=$1; shift
STORIES=("$@")
[ ${#STORIES[@]} -eq 0 ] && STORIES=(wheretheressmoke adollshouse avatar buck exorcism food legacy naked souls sweetaspie treasureisland)

ok() { "$PY" -I -c 'import sys,h5py; f=h5py.File(sys.argv[1],"r"); print(sys.argv[1].rsplit("/",1)[1], {k: f[k].shape for k in f})' "$1"; }

mkdir -p "$DEST/$SUB"
for s in "${STORIES[@]}"; do
  f="$DEST/$SUB/$s.hf5"
  if [ -f "$f" ] && ok "$f" >/dev/null 2>&1; then echo "skip $SUB/$s"; continue; fi
  echo "get  $SUB/$s"
  aws s3 cp --no-sign-request --only-show-errors "$S3/$SUB/$s.hf5" "$f.part" && mv "$f.part" "$f"
  ok "$f"
done
du -sh "$DEST/$SUB"
