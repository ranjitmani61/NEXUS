#!/bin/bash
# C2 queue c2d (2026-11-10): prompt-lookup divergence diagnostic; strictly after queue c2c.
S=/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad
V=$S/venv/bin/python; B=$S/data/pg19/test/test; R=/home/user/NEXUS/results/c2
export PYTHONDONTWRITEBYTECODE=1
cd /home/user/NEXUS
L=$S/c2_queue.log; O=$S/c2_queue_out.log
until grep -qE "^\[[0-9:]{8}\] C2C QUEUE DONE$" $L; do sleep 30; done
step() { echo "[$(date -u +%T)] BEGIN $1" >> $L; shift; "$@" >> $O 2>&1; local rc=$?; echo "[$(date -u +%T)] END rc=$rc" >> $L; }
step "diag prompt-lookup divergence qwen 8K bf16+fp32" $V tests/c2_lookup_divergence.py --model-dir $S/models/qwen2.5-0.5b --book $B/860.txt --context 8192 --new-tokens 64 --output $R/diag_lookup_divergence_qwen05_T8192.json
echo "[$(date -u +%T)] C2D QUEUE DONE" >> $L
