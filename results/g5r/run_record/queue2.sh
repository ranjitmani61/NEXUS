#!/bin/bash
# G5-R follow-on queue (2026-11-04): re-runs of Granite configurations invalidated by the
# missing-hybrid-cache defect (see results/g5r/invalid/README.md). Starts after queue.sh exits.
S=/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad
V=$S/venv/bin/python; B=$S/data/pg19/test/test; R=/home/user/NEXUS/results/g5r
G=$S/models/granite-4.0-micro-base
F4="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt"
LIVE="nx_box quest:1024 quest:2048 quest:4096"
export PYTHONDONTWRITEBYTECODE=1
cd /home/user/NEXUS
filt() { grep --line-buffered -v -i "warn\|deprecat"; }
while pgrep -f "scratchpad/queue.sh" >/dev/null; do sleep 20; done
step() { echo "[$(date -u +%T)] BEGIN $1" >> $S/queue.log; shift; "$@" 2>&1 | filt >> $S/queue_out.log; echo "[$(date -u +%T)] END rc=${PIPESTATUS[0]}" >> $S/queue.log; }
step "Q2 quality granite 8K (rerun, fixed cache)" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 8192 --steps 32 --books $F4 --live $LIVE --raw-dir $S/raw --output $R/quality_granite4micro_T8192.json
step "Q2 quality granite 4K" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 4096 --steps 32 --books $F4 --live $LIVE --raw-dir $S/raw --output $R/quality_granite4micro_T4096.json
echo "[$(date -u +%T)] QUEUE2 DONE" >> $S/queue.log
