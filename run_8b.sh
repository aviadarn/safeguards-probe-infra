#!/usr/bin/env bash
# The same six arms on Qwen3-8B. Nothing in probeinfra changes; only MODEL does.
# 36 blocks x 4 poolings x 4096 hidden over 4,502 prompts = 5.5 GB of cache.
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
MODEL=Qwen/Qwen3-8B BS=4 LAYER=18 POOL=mean BOOT=600 ./run_all.sh
