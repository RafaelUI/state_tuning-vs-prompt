#!/bin/bash
# Состояние вместо системной строки на g1d 0.4B (20.10): учитель (кэш ответов один на оба плеча),
# два плеча старта (zero / fold), затем ответы на проверочные вопросы (жадно и 10 сэмплов).
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
M=$D/g1d_0.4b.safetensors; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt)
for init in zero fold; do
  [ -s $W/state_$init.safetensors ] || /usr/bin/time -l ${GEN_RUN:-gen-run}2 --model $M --vocab $V --system "$SYS" --distill \
    --train-questions $W/train_questions.txt --teacher-cache $W/teacher.json --teacher-samples 4 \
    --temp 1.0 --top-p 0.6 --seed 7 --max-tokens 160 --init $init --steps 300 --lr 3e-4 --batch 4 \
    --state-out $W/state_$init.safetensors > $W/distill_$init.log 2>&1
  echo "обучение $init rc=$? | $(awk '/peak memory footprint/ {printf "footprint %.2f ГБ", $1/1e9}' $W/distill_$init.log)"
  for s in greedy s10; do
    extra=""; [ $s = s10 ] && extra="--samples 10 --temp 1.0 --top-p 0.6 --seed 100"
    ${GEN_RUN:-gen-run}2 --model $M --vocab $V --questions $W/questions.txt --mode none --state-in $W/state_$init.safetensors \
      --max-tokens 120 $extra --out $W/g04_state_${init}_$s.json > $W/g04_state_${init}_$s.log 2>&1
    echo "ответы $init $s rc=$?"
  done
done
