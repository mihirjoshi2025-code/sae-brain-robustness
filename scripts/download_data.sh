#!/usr/bin/env bash
# Download a thin slice of ds003020 (LeBel/Huth) for one subject (UTS01).
# Preprocessed voxel matrices (.hf5) + word-timing transcripts (.TextGrid).
# Raw per-subject is 22GB; we grab ~10 train stories + 1 repeated test story.
set -euo pipefail

S3=s3://openneuro.org/ds003020
DEST=~/brain-llm-data/ds003020
SUB=UTS01
AWS="aws s3 cp --no-sign-request"

mkdir -p "$DEST/preprocessed/$SUB" "$DEST/textgrids"

# Repeated test story (multiple repeats -> noise ceiling)
TEST=wheretheressmoke
# Training stories (single-trial)
TRAIN=(adollshouse avatar buck exorcism food legacy naked souls sweetaspie treasureisland)

echo "[1/3] TextGrids (all, ~84MB)..."
$AWS --recursive "$S3/derivatives/TextGrids/" "$DEST/textgrids/" >/dev/null

echo "[2/3] Test story: $TEST ..."
$AWS "$S3/derivatives/preprocessed_data/$SUB/$TEST.hf5" "$DEST/preprocessed/$SUB/$TEST.hf5"

echo "[3/3] Training stories..."
for s in "${TRAIN[@]}"; do
  echo "   - $s"
  $AWS "$S3/derivatives/preprocessed_data/$SUB/$s.hf5" "$DEST/preprocessed/$SUB/$s.hf5"
done

echo "DONE. Contents:"
du -sh "$DEST"/* 2>/dev/null || true
