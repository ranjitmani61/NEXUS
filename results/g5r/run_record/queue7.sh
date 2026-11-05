#!/bin/bash
# G5-R queue v7 (after container restart) (2026-11-04), order fixed by contract Amendment A5. Unfiltered logs.
S=/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad
V=$S/venv/bin/python; B=$S/data/pg19/test/test; R=/home/user/NEXUS/results/g5r
Q=$S/models/qwen2.5-0.5b; G=$S/models/granite-4.0-micro-base
ALL8="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt $B/3608.txt $B/3754.txt $B/4047.txt $B/5396.txt"
F4="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt"; F2="$B/860.txt $B/2544.txt"
LIVE="nx_box quest:1024 quest:2048 quest:4096"
export PYTHONDONTWRITEBYTECODE=1
cd /home/user/NEXUS
step() { echo "[$(date -u +%T)] BEGIN $1" >> $S/queue.log; shift; "$@" >> $S/queue_out.log 2>&1; echo "[$(date -u +%T)] END rc=$?" >> $S/queue.log; }
step "quality granite 8K (all fixes)" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 8192 --steps 32 --books $F4 --live $LIVE --raw-dir $S/raw --output $R/quality_granite4micro_T8192.json
step "timing granite bf16 b1 (native-GQA dense)" $V tests/g5r_timing.py --model-dir $G --model-name granite4micro --dtype bf16 --contexts 8192 16384 --methods dense quest:1024 quest:2048 nx_box nx_one --steps 8 --reps 3 --books $B/860.txt --output $R/timing_granite4micro_bf16_b1.json
step "decode consistency qwen BF16 8K (control for bf16 residual)" $V tests/g5r_decode_consistency.py --model-dir $Q --model-name qwen05_bf16 --dtype bf16 --books $F4 --T 8192 --output $R/decode_consistency_qwen05_bf16_T8192.json
step "decode consistency qwen 8K" $V tests/g5r_decode_consistency.py --model-dir $Q --model-name qwen05 --dtype fp32 --books $F4 --T 8192 --output $R/decode_consistency_qwen05_T8192.json
step "passkey granite 8K" $V tests/g5r_passkey.py --model-dir $G --model-name granite4micro --dtype bf16 --context 8192 --books $ALL8 --output $R/passkey_granite4micro_T8192.json
step "passkey qwen 32000" $V tests/g5r_passkey.py --model-dir $Q --model-name qwen05 --dtype fp32 --context 32000 --books $ALL8 --output $R/passkey_qwen05_T32000.json
step "quality granite 16K" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 16384 --steps 32 --books $F2 --live $LIVE --raw-dir $S/raw --output $R/quality_granite4micro_T16384.json
step "passkey qwen 8192" $V tests/g5r_passkey.py --model-dir $Q --model-name qwen05 --dtype fp32 --context 8192 --books $ALL8 --output $R/passkey_qwen05_T8192.json
step "quality granite 4K" $V tests/g5r_run_quality.py --model-dir $G --model-name granite4micro --dtype bf16 --context 4096 --steps 32 --books $F4 --live $LIVE --raw-dir $S/raw --output $R/quality_granite4micro_T4096.json
echo "[$(date -u +%T)] QUEUE7 DONE" >> $S/queue.log
