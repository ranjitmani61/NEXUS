#!/bin/bash
# C2 measurement queue v1 (2026-11-10). Charter: specs/C2_G0_batch1_decode_bottleneck_charter.md (+A1).
# Sequential; unfiltered logs; each step's rc recorded.
S=/tmp/claude-0/-home-user-NEXUS/892e3559-3d4b-5e4a-b712-9dfee2b2d122/scratchpad
V=$S/venv/bin/python; B=$S/data/pg19/test/test; R=/home/user/NEXUS/results/c2
Q=$S/models/qwen2.5-0.5b; G=$S/models/granite-4.0-micro-base
F4="$B/860.txt $B/2544.txt $B/3247.txt $B/3340.txt"
export PYTHONDONTWRITEBYTECODE=1
cd /home/user/NEXUS
L=$S/c2_queue.log; O=$S/c2_queue_out.log
step() { echo "[$(date -u +%T)] BEGIN $1" >> $L; shift; "$@" >> $O 2>&1; echo "[$(date -u +%T)] END rc=$?" >> $L; }
step "decode consistency qwen fp32 8K (host C)" $V tests/g5r_decode_consistency.py --model-dir $Q --model-name qwen05 --dtype fp32 --books $F4 --T 8192 --output $R/decode_consistency_qwen05_fp32_T8192_hostC.json
step "M1-M3 qwen bf16 breakdown" $V tests/c2_breakdown.py --model-dir $Q --model-name qwen05 --dtype bf16 --variants dense --contexts 4096 8192 16384 32000 --timing-book $B/860.txt --output $R/breakdown_qwen05_bf16.json
step "B1 qwen fp32 dense+int8 family, timing+quality" $V tests/c2_breakdown.py --model-dir $Q --model-name qwen05 --dtype fp32 --variants dense int8dyn int8dyn_pc int8dyn_pc_nolm --contexts 4096 16384 32000 --timing-book $B/860.txt --quality-books $F4 --quality-steps 32 --output $R/breakdown_b1_qwen05_fp32.json
step "decode consistency granite bf16 8K (host C)" $V tests/g5r_decode_consistency.py --model-dir $G --model-name granite4micro --dtype bf16 --books $F4 --T 8192 --output $R/decode_consistency_granite4micro_T8192_hostC.json
step "M1-M3 granite bf16 breakdown" $V tests/c2_breakdown.py --model-dir $G --model-name granite4micro --dtype bf16 --variants dense --contexts 4096 8192 16384 --timing-book $B/860.txt --output $R/breakdown_granite4micro_bf16.json
step "B2 qwen bf16 prompt lookup 8K" $V tests/c2_prompt_lookup.py --model-dir $Q --model-name qwen05 --dtype bf16 --context 8192 --new-tokens 64 --reps 3 --book $B/860.txt --output $R/b2_prompt_lookup_qwen05_bf16_T8192.json
step "B2 granite bf16 prompt lookup 8K (expected library refusal; recorded)" $V tests/c2_prompt_lookup.py --model-dir $G --model-name granite4micro --dtype bf16 --context 8192 --new-tokens 64 --reps 1 --book $B/860.txt --output $R/b2_prompt_lookup_granite4micro_bf16_T8192.json
echo "[$(date -u +%T)] C2 QUEUE DONE" >> $L
