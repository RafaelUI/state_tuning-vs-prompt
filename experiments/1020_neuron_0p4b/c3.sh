#!/bin/bash
# Серия c3 (20.10): данные c2 + 8 английских вопросов об имени, отброс пересказа системной строки
# (--reject-regex), старт со свёртки, 300 шагов, отсечение 1.0 (умолчание), 3 сида обучения (7, 8, 9).
# Общие способности: 12 ответов на вопрос у каждого плеча (состояние — 3 сида × 4 сэмпла; промпт и
# без промпта — 12 сэмплов); имя — 10 сэмплов на сид.
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
M=$D/g1d_0.4b.safetensors; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt); G=${GEN_RUN:-gen-run}5
for seed in 7 8 9; do
  arm=c3_fold_s$seed
  [ -s $W/$arm.safetensors ] || /usr/bin/time -l $G --model $M --vocab $V --system "$SYS" --distill \
    --train-questions $W/train_questions3.txt --teacher-cache $W/teacher3.json --teacher-samples 4 \
    --temp 1.0 --top-p 0.6 --max-tokens 160 --reject-regex "Ты — Neuron|System:" \
    --seed $seed --init fold --steps 300 --lr 3e-4 --batch 4 --state-out $W/$arm.safetensors > $W/distill_$arm.log 2>&1
  echo "$arm обучение rc=$? | $(grep -E '^учитель|^--reject|^в обучение' $W/distill_$arm.log | tr '\n' ' ') | $(grep -o 'шаг 300: лосс [0-9.]*' $W/distill_$arm.log)"
  $G --model $M --vocab $V --questions $W/questions.txt --mode none --state-in $W/$arm.safetensors \
    --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/g04_${arm}_s10.json > $W/g04_${arm}_s10.log 2>&1
  $G --model $M --vocab $V --questions $W/checks_q.txt --mode none --state-in $W/$arm.safetensors \
    --max-tokens 80 --samples 4 --temp 1.0 --top-p 0.6 --seed 200 --out $W/gen_${arm}_s4.json > $W/gen_${arm}_s4.log 2>&1
  echo "$arm ответы rc=$?"
done
for m in inline none; do
  [ -s $W/gen_${m}_s12.json ] || $G --model $M --vocab $V --questions $W/checks_q.txt --mode $m --system "$SYS" \
    --max-tokens 80 --samples 12 --temp 1.0 --top-p 0.6 --seed 200 --out $W/gen_${m}_s12.json > $W/gen_${m}_s12.log 2>&1
  echo "$m s12 rc=$?"
done
cd $W && python3 paired.py && python3 rates.py g04_inline_s10.json g04_c3_fold_s7_s10.json g04_c3_fold_s8_s10.json g04_c3_fold_s9_s10.json
