#!/bin/bash
# Против порчи фактов (20.10): к обучающим вопросам добавлены 40 фактов (только верные ответы учителя);
# старт zero и fold, 300 шагов; затем имя (questions.txt, 10 сэмплов) и общие способности (checks, жадно и 4 сэмпла).
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
M=$D/g1d_0.4b.safetensors; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt)
while ! grep -q "^EXIT" $HOME/Develop/swr_bench_keep/w1014/bg_general.log 2>/dev/null; do sleep 20; done
for init in zero fold; do
  arm=state2_$init
  [ -s $W/$arm.safetensors ] || /usr/bin/time -l ${GEN_RUN:-gen-run}3 --model $M --vocab $V --system "$SYS" --distill \
    --train-questions $W/train_questions2.txt --teacher-cache $W/teacher2.json --teacher-samples 4 \
    --temp 1.0 --top-p 0.6 --seed 7 --max-tokens 160 --init $init --steps 300 --lr 3e-4 --batch 4 \
    --state-out $W/$arm.safetensors > $W/distill2_$init.log 2>&1
  echo "обучение $init rc=$? | $(awk '/peak memory footprint/ {printf "footprint %.2f ГБ", $1/1e9}' $W/distill2_$init.log)"
  ${GEN_RUN:-gen-run}3 --model $M --vocab $V --questions $W/questions.txt --mode none --state-in $W/$arm.safetensors \
    --max-tokens 120 --samples 10 --temp 1.0 --top-p 0.6 --seed 100 --out $W/g04_${arm}_s10.json > $W/g04_${arm}_s10.log 2>&1
  for s in greedy s4; do
    extra=(); [ $s = s4 ] && extra=(--samples 4 --temp 1.0 --top-p 0.6 --seed 200)
    ${GEN_RUN:-gen-run}3 --model $M --vocab $V --questions $W/checks_q.txt --mode none --state-in $W/$arm.safetensors \
      --max-tokens 80 "${extra[@]}" --out $W/gen_${arm}_$s.json > $W/gen_${arm}_$s.log 2>&1
  done
  echo "ответы $arm rc=$?"
done
