#!/bin/bash
# 27.10: дымовая проверка артефактов g1d в gen-run (жадно, без промпта, 2 вопроса).
G=${SWIFTRWKV:-../SwiftRWKV}/.build/release/gen-run; V=models/rwkv_vocab_v20230424.txt; W=experiments/1027_matrix
for m in models/g1d_0.4b.safetensors $W/g1d_0p4b_reduction.rwkvq $W/g1d_0p4b_compression.rwkvq $W/g1d_1.5b.safetensors $W/g1d_1p5b_reduction.rwkvq $W/g1d_1p5b_compression.rwkvq; do
  echo "== $(basename $m)"
  /usr/bin/time -l $G --model $m --vocab $V --questions $W/smoke_q.txt --mode none --max-tokens 48 2>&1 | grep -v -E "^ +[0-9]+ +(voluntary|involuntary|instructions|cycles|block|page|messages|signals|swaps|average|maximum resident)" | tail -12
done
