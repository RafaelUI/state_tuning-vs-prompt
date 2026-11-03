#!/bin/bash
# Доли ответов с «Neuron»: 10 сэмплов на вопрос, temp 1.0, top-p 0.6; режимы none и inline.
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
SYS=$(cat $W/system.txt)
for m in none inline; do
  ${GEN_RUN:-gen-run} --model $D/g1d_0.4b.safetensors --vocab $D/rwkv_vocab_v20230424.txt --questions $W/questions.txt \
    --system "$SYS" --mode $m --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --max-tokens 120 \
    --out $W/g04_${m}_s10.json > $W/g04_${m}_s10.log 2>&1
  echo "$m rc=$?"
done
