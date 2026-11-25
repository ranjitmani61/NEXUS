#!/bin/bash
# G5-R queue v9 (replaces v8, whose sentinel matched its own log note) (2026-11-05): POST-HOC diagnostic of the Granite NX-BOX vs dense64 deviation
# (bf16 rounding amplification hypothesis). Not pre-registered; changes no decision rule.
# Starts only after queue v7 has finished, so the runs never compete for the 15 GB of RAM.
S=/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad
V=$S/venv/bin/python; B=$S/data/pg19/test/test; R=/home/user/NEXUS/results/g5r
Q=$S/models/qwen2.5-0.5b; G=$S/models/granite-4.0-micro-base
export PYTHONDONTWRITEBYTECODE=1
cd /home/user/NEXUS
until grep -qE "^\[[0-9:]{8}\] QUEUE7 DONE$" $S/queue.log; do sleep 60; done
step() { echo "[$(date -u +%T)] BEGIN $1" >> $S/queue.log; shift; "$@" >> $S/queue_out.log 2>&1; echo "[$(date -u +%T)] END rc=$?" >> $S/queue.log; }
step "diag granite bf16 amplification 4K" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 4096 --steps 32 --books $B/860.txt $B/2544.txt --live nx_box dense64p:1e-12 dense64p:1e-9 dense64p:1e-6 --capture-prob 0 --raw-dir $S/raw_diag --output $R/diag/bf16_amplification_granite4micro_T4096.json
step "diag qwen fp32 amplification 4K (negative control)" $V tests/g5r_run_quality.py --model-dir $Q --model-name qwen05 --dtype fp32 --context 4096 --steps 32 --books $B/860.txt $B/2544.txt --live nx_box dense64 dense64p:1e-6 --capture-prob 0 --raw-dir $S/raw_diag --output $R/diag/bf16_amplification_qwen05_fp32_T4096.json
echo "[$(date -u +%T)] QUEUE9 DONE" >> $S/queue.log
