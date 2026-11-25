#!/bin/bash
# C2 queue c2b (2026-11-10): B1 family per Amendment A2. Starts only after the main C2 queue's own completion line.
S=/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad
V=$S/venv/bin/python; B=$S/data/pg19/test/test; R=/home/user/NEXUS/results/c2
Q=$S/models/qwen2.5-0.5b; G=$S/models/granite-4.0-micro-base
F4="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt"
export PYTHONDONTWRITEBYTECODE=1
cd /home/user/NEXUS
L=$S/c2_queue.log; O=$S/c2_queue_out.log
until grep -qE "^\[[0-9:]{8}\] C2 QUEUE DONE$" $L; do sleep 30; done
step() { echo "[$(date -u +%T)] BEGIN $1" >> $L; shift; "$@" >> $O 2>&1; local rc=$?; echo "[$(date -u +%T)] END rc=$rc" >> $L; }
step "B1(A2) qwen fp32 dense+int8dyn family+int8wo" $V tests/c2_breakdown.py --model-dir $Q --model-name qwen05 --dtype fp32 --variants dense int8dyn int8dyn_pc int8dyn_pc_nolm int8wo --contexts 4096 16384 32000 --timing-book $B/860.txt --quality-books $F4 --quality-steps 32 --quality-only-if-faster --output $R/breakdown_b1_qwen05_fp32.json
step "B1(A2) qwen bf16 dense+int8wo" $V tests/c2_breakdown.py --model-dir $Q --model-name qwen05 --dtype bf16 --variants dense int8wo --contexts 4096 16384 32000 --timing-book $B/860.txt --quality-books $F4 --quality-steps 32 --quality-only-if-faster --output $R/breakdown_b1_qwen05_bf16.json
step "B1(A2) granite bf16 dense+int8wo (sequential variants)" $V tests/c2_breakdown.py --model-dir $G --model-name granite4micro --dtype bf16 --variants dense int8wo --contexts 4096 8192 16384 --timing-book $B/860.txt --quality-books $F4 --quality-steps 32 --quality-only-if-faster --sequential-variants --output $R/breakdown_b1_granite4micro_bf16.json
echo "[$(date -u +%T)] C2B QUEUE DONE" >> $L
