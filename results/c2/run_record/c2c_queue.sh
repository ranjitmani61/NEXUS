#!/bin/bash
# C2 queue c2c (2026-11-10): Qwen fp32 B1 rerun per Amendment A3.
S=/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad
V=$S/venv/bin/python; B=$S/data/pg19/test/test; R=/home/user/NEXUS/results/c2
Q=$S/models/qwen2.5-0.5b
F4="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt"
export PYTHONDONTWRITEBYTECODE=1
cd /home/user/NEXUS
L=$S/c2_queue.log; O=$S/c2_queue_out.log
step() { echo "[$(date -u +%T)] BEGIN $1" >> $L; shift; "$@" >> $O 2>&1; local rc=$?; echo "[$(date -u +%T)] END rc=$rc" >> $L; }
step "B1(A3) qwen fp32 dense+int8dyn family, sequential variants" $V tests/c2_breakdown.py --model-dir $Q --model-name qwen05 --dtype fp32 --variants dense int8dyn int8dyn_pc int8dyn_pc_nolm --contexts 4096 16384 32000 --timing-book $B/860.txt --quality-books $F4 --quality-steps 32 --quality-only-if-faster --sequential-variants --output $R/breakdown_b1_qwen05_fp32.json
echo "[$(date -u +%T)] C2C QUEUE DONE" >> $L
