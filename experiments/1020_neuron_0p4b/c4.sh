#!/bin/bash
# Серия c4 (21.10): данные c3 + 81 факт (train_questions4.txt, только верные ответы учителя), отброс пересказа
# системной строки, старт со свёртки, 300 шагов, отсечение 1.0 (умолчание), 3 сида обучения (7, 8, 9).
# Затем на каждом состоянии: общие на расширенном checks.tsv (199 вопросов, 4 сэмпла, сид 200) и имя (10 сэмплов).
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
M=$D/g1d_0.4b.safetensors; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt); G=${GEN_RUN:-gen-run}5
for seed in 7 8 9; do
  arm=c4_fold_s$seed
  [ -s $W/$arm.safetensors ] || /usr/bin/time -l $G --model $M --vocab $V --system "$SYS" --distill \
    --train-questions $W/train_questions4.txt --teacher-cache $W/teacher4.json --teacher-samples 4 \
    --temp 1.0 --top-p 0.6 --max-tokens 160 --reject-regex "Ты — Neuron|System:" \
    --seed $seed --init fold --steps 300 --lr 3e-4 --batch 4 --state-out $W/$arm.safetensors > $W/distill_$arm.log 2>&1
  echo "$arm обучение rc=$? | $(grep -E '^учитель|^--reject|^в обучение' $W/distill_$arm.log | tr '\n' ' ') | $(grep -o 'шаг 300: лосс [0-9.]*' $W/distill_$arm.log)"
done
for seed in 7 8 9; do
  arm=c4_fold_s$seed
  [ -s $W/g04_${arm}_s10.json ] || $G --model $M --vocab $V --questions $W/questions.txt --mode none --state-in $W/$arm.safetensors \
    --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/g04_${arm}_s10.json > $W/g04_${arm}_s10.log 2>&1
  [ -s $W/v2_${arm}_s4.json ] || $G --model $M --vocab $V --questions $W/checks_q.txt --mode none --state-in $W/$arm.safetensors \
    --max-tokens 80 --samples 4 --temp 1.0 --top-p 0.6 --seed 200 --out $W/v2_${arm}_s4.json > $W/v2_${arm}_s4.log 2>&1
  echo "$arm ответы rc=$? $(date '+%H:%M')"
done
