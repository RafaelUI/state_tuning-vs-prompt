#!/bin/bash
# 29.10, пункт 3: пять системных промптов на g1d 0.4B и 1.5B bf16 (тот же бинарник w1027/bin/gen-run, что матрица).
# Протокол c4: учитель — та же модель со строкой (4 сэмпла, temp 1.0, top-p 0.6, сид 7), фильтр — вид fact
# (p/<промпт>/train.txt: соблюдение промпта И верный факт), отброс пересказа промпта; fold, 300 шагов, lr 3e-4,
# батч 4, отсечение 1.0, сиды 7/8/9. Проверка: общие (checks_q, промпт 12 сэмплов, состояние 4 × 3 сида, сид 200,
# 80 токенов); у «orion» ещё имя (questions.txt, 10). «Без промпта» — из матрицы (w1027/m/<модель>, тот же бинарник).
W=experiments/1029_prompts; N=data/neuron
V=models/rwkv_vocab_v20230424.txt; G=${GEN_RUN:-gen-run}
model_path() { case $1 in g1d_0p4b_bf16) echo models/g1d_0.4b.safetensors;; g1d_1p5b_bf16) echo models/g1d_1.5b.safetensors;; esac; }
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
one() {  # one <промпт> <модель>
  local p=$1 tag=$2 M=$(model_path $2) O=$W/p/$1/$2; mkdir -p $O
  local SYS; SYS=$(cat $W/p/$p/system.txt); local REJ; REJ=$(cat $W/p/$p/reject.txt)
  echo "### $p / $tag старт $(date '+%F %H:%M') своп $(sw)"
  for s in 7 8 9; do
    [ -s $O/c4_s$s.safetensors ] && continue
    /usr/bin/time -l $G --model $M --vocab $V --system "$SYS" --distill --train-questions $W/p/$p/train.txt \
      --teacher-cache $O/teacher.json --teacher-samples 4 --temp 1.0 --top-p 0.6 --max-tokens 160 \
      --reject-regex "$REJ" --seed $s --init fold --steps 300 --lr 3e-4 --batch 4 \
      --state-out $O/c4_s$s.safetensors > $O/distill_c4_s$s.log 2>&1
    echo "  $p/$tag c4_s$s rc=$? | $(grep -E '^учитель|^--reject|^в обучение' $O/distill_c4_s$s.log | tr '\n' ' ' | cut -c1-260) | $(grep -o 'шаг 300: лосс [0-9.]*' $O/distill_c4_s$s.log) | пик $(grep 'peak memory footprint' $O/distill_c4_s$s.log | awk '{printf "%.2f ГБ", $1/1e9}') | $(date '+%H:%M')"
  done
  gen() { local t=$1 q=$2 n=$3 k=$4 sd=$5; shift 5; [ -s $O/$t.json ] && return
    $G --model $M --vocab $V --questions $N/$q --system "$SYS" "$@" --max-tokens $k --samples $n \
      --temp 1.0 --top-p 0.6 --seed $sd --out $O/$t.json > $O/$t.log 2>&1; echo "  $p/$tag $t rc=$? $(date '+%H:%M')"; }
  gen v2_inline_s12 checks_q.txt 12 80 200 --mode inline
  for s in 7 8 9; do gen v2_c4_s${s}_s4 checks_q.txt 4 80 200 --mode none --state-in $O/c4_s$s.safetensors; done
  if [ $p = orion ]; then
    gen id_inline_s10 questions.txt 10 120 100 --mode inline
    for s in 7 8 9; do gen id_c4_s${s}_s10 questions.txt 10 120 100 --mode none --state-in $O/c4_s$s.safetensors; done
  fi
  python3 $W/prompts.py score $W/p $p $tag > /dev/null 2>&1 && echo "  сводка: $O/summary.md"
  echo "### $p / $tag конец $(date '+%F %H:%M') своп $(sw)"
}
echo "== промпты старт $(date '+%F %T') своп $(sw); ${GEN_RUN:-gen-run}"
for p in ${PROMPTS:-orion plain short english json}; do for m in g1d_0p4b_bf16 g1d_1p5b_bf16; do one $p $m; done; done
echo "== промпты конец $(date '+%F %T') своп $(sw)"
