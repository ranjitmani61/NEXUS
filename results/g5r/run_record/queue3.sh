#!/bin/bash
# G5-R queue v3 (2026-11-04), order fixed by contract Amendment A3.
S=/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad
V=$S/venv/bin/python; B=$S/data/pg19/test/test; R=/home/user/NEXUS/results/g5r
Q=$S/models/qwen2.5-0.5b; G=$S/models/granite-4.0-micro-base
ALL8="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt $B/3608.txt $B/3754.txt $B/4047.txt $B/5396.txt"
F4="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt"
F2="$B/860.txt $B/2544.txt"
LIVE="nx_box quest:1024 quest:2048 quest:4096"
export PYTHONDONTWRITEBYTECODE=1
cd /home/user/NEXUS
filt() { grep --line-buffered -v -i "warn\|deprecat"; }
while pgrep -f "g5r_run_quality.py.*--context 8192" >/dev/null; do sleep 15; done
echo "[$(date -u +%T)] END (qwen 8K, finished under queue v1 controller)" >> $S/queue.log
step() { echo "[$(date -u +%T)] BEGIN $1" >> $S/queue.log; shift; "$@" 2>&1 | filt >> $S/queue_out.log; echo "[$(date -u +%T)] END rc=${PIPESTATUS[0]}" >> $S/queue.log; }
step "timing qwen bf16 b4" $V tests/g5r_timing.py --model-dir $Q --model-name qwen05 --dtype bf16 --contexts 8192 16384 32000 --methods dense quest:1024 quest:2048 nx_box nx_one --steps 16 --reps 5 --batch 4 --books $F4 --output $R/timing_qwen05_bf16_b4.json
step "quality granite 8K (fixed cache)" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 8192 --steps 32 --books $F4 --live $LIVE --raw-dir $S/raw --output $R/quality_granite4micro_T8192.json
step "timing granite bf16 b1" $V tests/g5r_timing.py --model-dir $G --model-name granite4micro --dtype bf16 --contexts 8192 16384 --methods dense quest:1024 quest:2048 quest:4096 nx_box nx_sph nx_one --steps 16 --reps 5 --books $B/860.txt --output $R/timing_granite4micro_bf16_b1.json
step "quality qwen 16K" $V tests/g5r_run_quality.py --model-dir $Q --model-name qwen05 --dtype fp32 --context 16384 --steps 32 --books $F4 --live $LIVE --raw-dir $S/raw --output $R/quality_qwen05_T16384.json
step "quality qwen 32K" $V tests/g5r_run_quality.py --model-dir $Q --model-name qwen05 --dtype fp32 --context 32000 --steps 32 --books $F2 --live $LIVE --raw-dir $S/raw --output $R/quality_qwen05_T32000.json
for T in 16384 32000 8192; do step "passkey qwen $T" $V tests/g5r_passkey.py --model-dir $Q --model-name qwen05 --dtype fp32 --context $T --books $ALL8 --output $R/passkey_qwen05_T$T.json; done
step "passkey granite 8K" $V tests/g5r_passkey.py --model-dir $G --model-name granite4micro --dtype bf16 --context 8192 --books $ALL8 --output $R/passkey_granite4micro_T8192.json
step "adversarial qwen random 4K" $V tests/g5r_run_quality.py --model-dir $Q --model-name qwen05 --dtype fp32 --context 4096 --steps 32 --books $F2 --input-kind random --live $LIVE --raw-dir $S/raw_adv --output $R/quality_qwen05_T4096_random.json
step "adversarial qwen repeat 4K" $V tests/g5r_run_quality.py --model-dir $Q --model-name qwen05 --dtype fp32 --context 4096 --steps 32 --books $F2 --input-kind repeat --live $LIVE --raw-dir $S/raw_adv --output $R/quality_qwen05_T4096_repeat.json
step "quality granite 16K" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 16384 --steps 32 --books $F2 --live $LIVE --raw-dir $S/raw --output $R/quality_granite4micro_T16384.json
step "quality granite 4K" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 4096 --steps 32 --books $F4 --live $LIVE --raw-dir $S/raw --output $R/quality_granite4micro_T4096.json
echo "[$(date -u +%T)] QUEUE3 DONE" >> $S/queue.log
