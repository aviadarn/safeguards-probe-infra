#!/usr/bin/env bash
# Everything in the README, end to end. One GPU pass per arm; all analysis is CPU.
#
# MODEL is the only thing worth changing. The six arms below are not six
# experiments — arms 1-4 are the real evaluation, and arms 5-6 exist only to
# price the pooling bug under the two settings that control it.
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
BS="${BS:-8}"
BOOT="${BOOT:-600}"
LAYER="${LAYER:-16}"
POOL="${POOL:-mean}"
E="python -m probeinfra extract --model $MODEL --batch-size $BS"

echo "=== 1-4. the evaluation: one pass per prompt set, left padding, length-sorted"
for S in wildguard-vanilla wildguard-adversarial toxicchat-toxicity toxicchat-jailbreaking; do
  $E --set "$S"
done

echo "=== 5-6. pricing the pooling bug: right padding, then right padding unsorted"
$E --set wildguard-vanilla --padding-side right
$E --set wildguard-vanilla --padding-side right --no-sort

echo "=== sweeps (CPU only) ==="
python -m probeinfra sweep --model "$MODEL" --set wildguard-vanilla --boot "$BOOT"
python -m probeinfra sweep --model "$MODEL" --set wildguard-vanilla --padding-side right --boot "$BOOT"
python -m probeinfra sweep --model "$MODEL" --set wildguard-vanilla --padding-side right --no-sort --boot "$BOOT"

echo "=== controls (exits non-zero if any control fails) ==="
python -m probeinfra controls --model "$MODEL" --set wildguard-vanilla \
    --layer "$LAYER" --pooling "$POOL" --boot "$BOOT"

echo "=== generalisation: fit once on plain prompts, score the rest ==="
python -m probeinfra transfer --model "$MODEL" --layer "$LAYER" --pooling "$POOL" \
    --to wildguard-adversarial,toxicchat-toxicity,toxicchat-jailbreaking --boot "$BOOT"
