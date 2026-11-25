#!/bin/bash
# G5-R follow-on queue (2026-11-04): Granite 8K re-run after the RoPE-offset fix; standing decode checks.
S=/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad
V=$S/venv/bin/python; B=$S/data/pg19/test/test; R=/home/user/NEXUS/results/g5r
Q=$S/models/qwen2.5-0.5b; G=$S/models/granite-4.0-micro-base
F4="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt"
LIVE="nx_box quest:1024 quest:2048 quest:4096"
export PYTHONDONTWRITEBYTECODE=1
cd /home/user/NEXUS
while pgrep -f "scratchpad/queue4.sh" >/dev/null; do sleep 20; done
step() { echo "[$(date -u +%T)] BEGIN $1" >> $S/queue.log; shift; "$@" >> $S/queue_out.log 2>&1; echo "[$(date -u +%T)] END rc=$?" >> $S/queue.log; }
step "Q5 decode consistency qwen 8K" $V tests/g5r_decode_consistency.py --model-dir $Q --model-name qwen05 --dtype fp32 --books $F4 --T 8192 --output $R/decode_consistency_qwen05_T8192.json
step "Q5 quality granite 8K (RoPE fix)" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 8192 --steps 32 --books $F4 --live $LIVE --raw-dir $S/raw --output $R/quality_granite4micro_T8192.json
step "Q5 decode consistency granite 8K (with ulp fields)" $V tests/g5r_decode_consistency.py --model-dir $G --model-name granite4micro --dtype bf16 --books $F4 --T 8192 --output $R/decode_consistency_granite4micro_T8192.json
step "Q5 timing granite bf16 b1 (native-GQA dense baseline)" $V tests/g5r_timing.py --model-dir $G --model-name granite4micro --dtype bf16 --contexts 8192 16384 --methods dense quest:1024 quest:2048 nx_box nx_one --steps 8 --reps 3 --books $B/860.txt --output $R/timing_granite4micro_bf16_b1.json
echo "[$(date -u +%T)] QUEUE5 DONE" >> $S/queue.log
