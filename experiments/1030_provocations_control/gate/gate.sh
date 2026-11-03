#!/bin/bash
# 30.10: гейт нового gen-run (--train/--top-in/--lora-in): на старых плечах матрицы ответы побитно равны прежним.
set -e
cd ${SWIFTRWKV:-../SwiftRWKV} && swift build -c release --product gen-run 2>&1 | grep -E "error|Build complete"
G=${SWIFTRWKV:-../SwiftRWKV}/.build/release/gen-run; W=experiments/1030_provocations_control/gate; N=data/neuron
M=models/g1d_0.4b.safetensors; V=models/rwkv_vocab_v20230424.txt; O=experiments/1027_matrix/m/g1d_0p4b_bf16
SYS=$(cat $N/system.txt)
$G --model $M --vocab $V --questions $N/questions.txt --system "$SYS" --mode none --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/id_none_s10.json > $W/id_none.log 2>&1
$G --model $M --vocab $V --questions $N/checks_q.txt --system "$SYS" --mode none --state-in $O/c4_s7.safetensors --max-tokens 80 --samples 4 --temp 1.0 --top-p 0.6 --seed 200 --out $W/v2_c4_s7_s4.json > $W/v2.log 2>&1
$G --model $M --vocab $V --questions $N/questions.txt --system "$SYS" --mode inline --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/id_inline_s10.json > $W/id_inline.log 2>&1
python3 - <<'PY'
import json
W='experiments/1030_provocations_control/gate'; O='experiments/1027_matrix/m/g1d_0p4b_bf16'
for f in ['id_none_s10.json','v2_c4_s7_s4.json','id_inline_s10.json']:
    a=json.load(open(f'{W}/{f}'))['rows']; b=json.load(open(f'{O}/{f}'))['rows']
    print(f, 'РАВНО' if a==b else 'РАЗНЫЕ', sum(x!=y for x,y in zip(a,b)), 'строк отличается из', len(a))
PY
