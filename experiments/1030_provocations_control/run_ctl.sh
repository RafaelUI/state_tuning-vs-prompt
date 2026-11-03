#!/bin/bash
# 30.10, пункт 2: сильный контроль к обучению состояния (Neuron, данные c4, тот же кэш учителя, что у матрицы).
# Учатся веса без системной строки: top — верхние 4 слоя + ln_out; lora — адаптеры r16 на tmix всех слоёв.
# Расписание как у состояния (cosine, lrMin = lr/10, β₂ 0.99, отсечение 1.0, 300 шагов, батч 4).
# Фаза A: сид 7 × 3 LR, выбор по валидации ctl/<модель>/valall.tsv (факты вне обучения: у учителя 0 верных + 40 новых).
# Фаза B: лучший LR × сиды 7/8/9; проверка как у матрицы: общие 199 (s4, сид 200), имя (s10, сид 100),
# провокации (s10, сид 300, 120 токенов), перенос (s4, сид 200). Бинарник — SwiftRWKV с --train (коммит в журнале).
# --state-out для --train top|lora не используется, но 3106b2d всё ещё требует его в --distill (заглушка).
W=experiments/1030_provocations_control; N=data/neuron; B=experiments/1027_matrix
V=models/rwkv_vocab_v20230424.txt; G=${GEN_RUN:-gen-run}; PY=python3
SYS=$(cat $N/system.txt)
path() { case $1 in g1d_0p4b_bf16) echo models/g1d_0.4b.safetensors;; g1d_1p5b_bf16) echo $B/g1d_1.5b.safetensors;; esac; }
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
train() {  # train <модель> <вид> <lr> <сид>
  local m=$1 k=$2 lr=$3 s=$4 O=$W/ctl/$1; local f=$O/${k}_lr${lr}_s$s.safetensors
  [ -s $f ] && return
  /usr/bin/time -l $G --model $(path $m) --vocab $V --system "$SYS" --distill --train-questions $N/train_questions4.txt \
    --teacher-cache $O/teacher.json --teacher-samples 4 --temp 1.0 --top-p 0.6 --max-tokens 160 \
    --reject-regex "Ты — Neuron|System:" --seed $s --steps 300 --lr $lr --batch 4 --train $k --weights-out $f \
    --state-out $O/unused_state.safetensors > $O/train_${k}_lr${lr}_s$s.log 2>&1
  local rc=$?
  [ -s $f ] || { echo "ОШИБКА обучения $m $k lr $lr s$s (rc $rc): $(head -3 $O/train_${k}_lr${lr}_s$s.log | tail -2)"; exit 3; }
  echo "  $m $k lr $lr s$s обучение rc=$rc | $(grep -E '^в обучение' $O/train_${k}_lr${lr}_s$s.log | cut -c1-60) | $(grep -E 'сохранен' $O/train_${k}_lr${lr}_s$s.log | grep -o 'лосс [0-9.]*, [0-9]* с') | пик $(grep 'peak memory footprint' $O/train_${k}_lr${lr}_s$s.log | awk '{printf "%.2f ГБ", $1/1e9}') | своп $(sw) | $(date '+%H:%M')"
}
wflag() { case $1 in top) echo --top-in;; lora) echo --lora-in;; esac; }
gen() {  # gen <модель> <выход> <вопросы> <сэмплов> <токенов> <сид> <аргументы…>
  local m=$1 o=$2 q=$3 n=$4 k=$5 sd=$6; shift 6; [ -s $o ] && return
  $G --model $(path $m) --vocab $V --questions $q --system "$SYS" --mode none "$@" --max-tokens $k --samples $n \
    --temp 1.0 --top-p 0.6 --seed $sd --out $o > ${o%.json}.log 2>&1; echo "  $(basename $o) rc=$? $(date '+%H:%M')"
}
echo "== контроль старт $(date '+%F %T') своп $(sw); ${GEN_RUN:-gen-run}"
for spec in "g1d_0p4b_bf16 top 3e-5 1e-4 3e-4" "g1d_1p5b_bf16 lora 1e-4 3e-4 1e-3" "g1d_1p5b_bf16 top 1e-5 3e-5 1e-4"; do
  set -- $spec; m=$1; k=$2; shift 2; O=$W/ctl/$m
  echo "### $m $k: фаза A (сид 7, LR $*)"
  # опорные точки валидации: без промпта и состояние c4 сид 7 (тот же бинарник)
  gen $m $O/val_none.json $O/valall_q.txt 4 80 200
  gen $m $O/val_c4_s7.json $O/valall_q.txt 4 80 200 --state-in $B/m/$m/c4_s7.safetensors
  best=""; bestv=-1
  for lr in "$@"; do
    train $m $k $lr 7
    gen $m $O/val_${k}_lr${lr}_s7.json $O/valall_q.txt 4 80 200 $(wflag $k) $O/${k}_lr${lr}_s7.safetensors
    v=$($PY $W/ctl_val.py score $O/valall.tsv $O/val_${k}_lr${lr}_s7.json)
    echo "  валидация $k lr $lr: $v (без промпта $($PY $W/ctl_val.py score $O/valall.tsv $O/val_none.json), c4 $($PY $W/ctl_val.py score $O/valall.tsv $O/val_c4_s7.json))"
    if $PY -c "import sys; sys.exit(0 if float('$v') > float('$bestv') else 1)"; then best=$lr; bestv=$v; fi
  done
  echo "### $m $k: выбран lr $best (валидация $bestv); фаза B"
  echo "$best" > $O/${k}_best_lr.txt
  for s in 7 8 9; do
    train $m $k $best $s; f=$O/${k}_lr${best}_s$s.safetensors
    gen $m $O/v2_${k}_s${s}_s4.json $N/checks_q.txt 4 80 200 $(wflag $k) $f
    gen $m $O/id_${k}_s${s}_s10.json $N/questions.txt 10 120 100 $(wflag $k) $f
    gen $m $O/prov_${k}_s${s}.json $W/provocations.txt 10 120 300 $(wflag $k) $f
    gen $m $O/v3_${k}_s${s}_s4.json $N/transfer_q.txt 4 80 200 $(wflag $k) $f
  done
done
echo "== контроль конец $(date '+%F %T') своп $(sw)"
