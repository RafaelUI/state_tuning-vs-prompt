#!/bin/bash
# 01.11: гейт правки «GPU.clearCache() после каждого хода диалога» (только замыкание --dialogues).
# Новый бинарник воспроизводит готовые плечи прогона mt (0.4B: промпт, c4 сид 7) побитно; свойства CLI.
set -e
G=${SWIFTRWKV:-../SwiftRWKV}/.build/release/gen-run; W=experiments/1101_dialogues; N=data/neuron
M=models/g1d_0.4b.safetensors; V=models/rwkv_vocab_v20230424.txt; SYS=$(cat $N/system.txt)
/usr/bin/time -l $G --model $M --vocab $V --dialogues $W/dialogues.json --system "$SYS" --mode inline --max-tokens 120 --samples 8 --temp 1.0 --top-p 0.6 --seed 500 --out $W/gate/g2_inline.json > $W/gate/g2_inline.log 2>&1
/usr/bin/time -l $G --model $M --vocab $V --dialogues $W/dialogues.json --system "$SYS" --mode none --state-in experiments/1027_matrix/m/g1d_0p4b_bf16/c4_s7.safetensors --max-tokens 120 --samples 8 --temp 1.0 --top-p 0.6 --seed 500 --out $W/gate/g2_c4_s7.json > $W/gate/g2_c4.log 2>&1
for a in inline c4; do echo "$a: пик $(grep 'peak memory footprint' $W/gate/g2_$a.log | awk '{printf "%.2f ГБ", $1/1e9}'), $(grep -o 'всего [0-9.]* с' $W/gate/g2_$a.log)"; done
/usr/bin/python3 - <<'PY'
import json
W='experiments/1101_dialogues'; ok=True
for a,b in (('gate/g2_inline.json','mt/g1d_0p4b_bf16/inline.json'),('gate/g2_c4_s7.json','mt/g1d_0p4b_bf16/c4_s7.json')):
    e=json.load(open(f'{W}/{a}'))['dialogues']==json.load(open(f'{W}/{b}'))['dialogues']; ok&=e; print(b, 'ПОБИТНО РАВНО' if e else 'РАЗНОЕ')
print('ГЕЙТ', 'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН'); raise SystemExit(0 if ok else 1)
PY
cd ${SWIFTRWKV:-../SwiftRWKV} && ./Scripts/check_genrun_cli.sh
