#!/bin/bash
# v3arm.sh <none|inline|c4_fold_sN> — перенос формы (22.10): ответы на transfer_q.txt (80 вопросов: 40 единиц ×
# шаблон обучения / новая формулировка). Без промпта и промпт — 12 сэмплов, состояние — 4; temp 1.0, top-p 0.6,
# сид 200, до 80 токенов (как в v2arm.sh).
W=$HOME/Develop/swr_bench_keep/w1020/neuron; D=$HOME/Develop/swr_bench_keep/w1014
M=$D/g1d_0.4b.safetensors; V=$D/rwkv_vocab_v20230424.txt; SYS=$(cat $W/system.txt); G=${GEN_RUN:-gen-run}5
arm=$1
case $arm in
  none|inline) out=$W/v3_${arm}_s12.json; extra=(--mode $arm --samples 12) ;;
  *) out=$W/v3_${arm}_s4.json; extra=(--mode none --state-in $W/$arm.safetensors --samples 4) ;;
esac
[ -s $out ] && { echo "$arm | уже есть"; exit 0; }
$G --model $M --vocab $V --questions $W/transfer_q.txt --system "$SYS" "${extra[@]}" \
  --max-tokens 80 --temp 1.0 --top-p 0.6 --seed 200 --out $out > ${out%.json}.log 2>&1
echo "$arm rc=$? $(date '+%d.%m %H:%M')"
