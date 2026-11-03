#!/bin/bash
# Расширенный проверочный набор (21.10, checks.tsv — 199 вопросов + имя): без промпта и промпт — 12 сэмплов
# (сид 200, как в c3), затем прежние состояния c3 (3 сида × 4 сэмпла) — для разделения «больше фактов в обучении»
# и «больше вопросов в проверке».
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
M=$D/g1d_0.4b.safetensors; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt); G=${GEN_RUN:-gen-run}5
for m in none inline; do
  [ -s $W/v2_${m}_s12.json ] || $G --model $M --vocab $V --questions $W/checks_q.txt --mode $m --system "$SYS" \
    --max-tokens 80 --samples 12 --temp 1.0 --top-p 0.6 --seed 200 --out $W/v2_${m}_s12.json > $W/v2_${m}_s12.log 2>&1
  echo "$m s12 rc=$? $(date '+%H:%M')"
done
for seed in 7 8 9; do
  arm=c3_fold_s$seed
  [ -s $W/v2_${arm}_s4.json ] || $G --model $M --vocab $V --questions $W/checks_q.txt --mode none --state-in $W/$arm.safetensors \
    --max-tokens 80 --samples 4 --temp 1.0 --top-p 0.6 --seed 200 --out $W/v2_${arm}_s4.json > $W/v2_${arm}_s4.log 2>&1
  echo "$arm v2 rc=$? $(date '+%H:%M')"
done
