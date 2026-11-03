#!/bin/bash
# Общие способности (20.10): 39 вопросов с проверяемым ответом + имя; плечи none, inline,
# state_zero, state_fold; жадно и 4 сэмпла (temp 1.0, top-p 0.6).
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
M=$D/g1d_0.4b.safetensors; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt)
for arm in none inline state_zero state_fold; do
  for s in greedy s4; do
    extra=(); [ $s = s4 ] && extra=(--samples 4 --temp 1.0 --top-p 0.6 --seed 200)
    case $arm in
      none|inline) how=(--mode $arm --system "$SYS") ;;
      *) how=(--mode none --state-in $W/$arm.safetensors) ;;
    esac
    ${GEN_RUN:-gen-run}2 --model $M --vocab $V --questions $W/checks_q.txt "${how[@]}" --max-tokens 80 "${extra[@]}" \
      --out $W/gen_${arm}_$s.json > $W/gen_${arm}_$s.log 2>&1
    echo "$arm $s rc=$?"
  done
done
cd $W && python3 score.py gen_none_greedy.json gen_inline_greedy.json gen_state_zero_greedy.json gen_state_fold_greedy.json
python3 score.py gen_none_s4.json gen_inline_s4.json gen_state_zero_s4.json gen_state_fold_s4.json
