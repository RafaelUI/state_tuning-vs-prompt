#!/bin/bash
# Досчёт серии c3 после случайной остановки (21.10): только прогон без промпта (12 сэмплов) и сводки.
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
[ -s $W/gen_none_s12.json ] || ${GEN_RUN:-gen-run}5 --model $D/g1d_0.4b.safetensors --vocab $D/rwkv_vocab_v20230424.txt \
  --questions $W/checks_q.txt --mode none --max-tokens 80 --samples 12 --temp 1.0 --top-p 0.6 --seed 200 \
  --out $W/gen_none_s12.json > $W/gen_none_s12.log 2>&1
echo "none s12 rc=$?"
cd $W && python3 paired.py && python3 rates.py g04_inline_s10.json g04_c3_fold_s7_s10.json g04_c3_fold_s8_s10.json g04_c3_fold_s9_s10.json
