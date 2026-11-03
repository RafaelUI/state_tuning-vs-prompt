#!/bin/bash
# Отсечение 10 против 1 (20.10) на конфигурации, которая без отсечения срывалась (c2, старт с нуля, шаг 13).
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
M=$D/g1d_0.4b.safetensors; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt)
arm=c2_zero_clip10
[ -s $W/$arm.safetensors ] || ${GEN_RUN:-gen-run}4 --model $M --vocab $V --system "$SYS" --distill \
  --train-questions $W/train_questions2.txt --teacher-cache $W/teacher2.json --seed 7 --init zero --steps 300 \
  --lr 3e-4 --batch 4 --grad-clip 10 --log-every 1 --state-out $W/$arm.safetensors > $W/distill_$arm.log 2>&1
echo "$arm обучение rc=$? | $(grep -o 'шаг 300: лосс [0-9.]*' $W/distill_$arm.log)"
${GEN_RUN:-gen-run}4 --model $M --vocab $V --questions $W/questions.txt --mode none --state-in $W/$arm.safetensors \
  --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/g04_${arm}_s10.json > $W/g04_${arm}_s10.log 2>&1
for s in greedy s4; do
  extra=(); [ $s = s4 ] && extra=(--samples 4 --temp 1.0 --top-p 0.6 --seed 200)
  ${GEN_RUN:-gen-run}4 --model $M --vocab $V --questions $W/checks_q.txt --mode none --state-in $W/$arm.safetensors \
    --max-tokens 80 "${extra[@]}" --out $W/gen_${arm}_$s.json > $W/gen_${arm}_$s.log 2>&1
done
echo "$arm ответы rc=$?"
cd $W && python3 score.py gen_c2_zero_greedy.json gen_c2_zero_clip10_greedy.json | head -2
python3 score.py gen_c2_zero_s4.json gen_c2_zero_clip10_s4.json | head -2
python3 rates.py g04_c2_zero_s10.json g04_c2_zero_clip10_s10.json
