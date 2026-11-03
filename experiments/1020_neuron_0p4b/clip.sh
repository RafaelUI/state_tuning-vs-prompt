#!/bin/bash
# Чистая серия (20.10): отсечение градиента 1.0 (без него старт с нуля срывается на шаге 13:
# лосс 1.03 → 10.98, норма 233). Данные: c1 — train_questions.txt (без фактов), c2 —
# train_questions2.txt (+40 фактов, только верные ответы учителя); старт zero / fold; 300 шагов.
# Затем имя (questions.txt, 10 сэмплов) и общие способности (checks, жадно и 4 сэмпла).
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
M=$D/g1d_0.4b.safetensors; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt)
for data in c1 c2; do
  if [ $data = c1 ]; then tq=train_questions.txt; tc=teacher.json; else tq=train_questions2.txt; tc=teacher2.json; fi
  for init in zero fold; do
    arm=${data}_$init
    [ -s $W/$arm.safetensors ] || /usr/bin/time -l ${GEN_RUN:-gen-run}4 --model $M --vocab $V --system "$SYS" --distill \
      --train-questions $W/$tq --teacher-cache $W/$tc --seed 7 --init $init --steps 300 --lr 3e-4 --batch 4 \
      --grad-clip 1.0 --state-out $W/$arm.safetensors > $W/distill_$arm.log 2>&1
    echo "$arm обучение rc=$? | $(awk '/peak memory footprint/ {printf "footprint %.2f ГБ", $1/1e9}' $W/distill_$arm.log) | $(grep -o 'шаг 300: лосс [0-9.]*' $W/distill_$arm.log)"
    ${GEN_RUN:-gen-run}4 --model $M --vocab $V --questions $W/questions.txt --mode none --state-in $W/$arm.safetensors \
      --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/g04_${arm}_s10.json > $W/g04_${arm}_s10.log 2>&1
    for s in greedy s4; do
      extra=(); [ $s = s4 ] && extra=(--samples 4 --temp 1.0 --top-p 0.6 --seed 200)
      ${GEN_RUN:-gen-run}4 --model $M --vocab $V --questions $W/checks_q.txt --mode none --state-in $W/$arm.safetensors \
        --max-tokens 80 "${extra[@]}" --out $W/gen_${arm}_$s.json > $W/gen_${arm}_$s.log 2>&1
    done
    echo "$arm ответы rc=$?"
  done
done
cd $W && python3 score.py gen_none_greedy.json gen_inline_greedy.json gen_c1_zero_greedy.json gen_c1_fold_greedy.json gen_c2_zero_greedy.json gen_c2_fold_greedy.json
python3 score.py gen_none_s4.json gen_inline_s4.json gen_c1_zero_s4.json gen_c1_fold_s4.json gen_c2_zero_s4.json gen_c2_fold_s4.json
python3 rates.py g04_inline_s10.json g04_c1_zero_s10.json g04_c1_fold_s10.json g04_c2_zero_s10.json g04_c2_fold_s10.json
