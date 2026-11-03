#!/bin/bash
# 01.11: классификация новостей (Lenta.ru, 5 рубрик) на g1d 0.4B bf16 — следование инструкции вместо роли (решение Алексея 01.11:
# у 0.4B не хватает глубины на многоходовую роль). Данные — build.py, build_val.py.
# Учитель 0.4B инструкцию не выполняет ни в каком виде (системная строка: 0 из 1000 ответов «ровно рубрика»; инструкция после
# документа и 5 примеров — 0 из 60, probe/), поэтому дистилляции нет (плечо cd снято). Решение Алексея 01.11: золотые метки +
# сильный контроль. Плечи: без промпта; промпт (инструкция в строке); cg — состояние на золотых метках (кэш teacher_g.json,
# протокол c4: fold, 300 шагов, lr 3e-4, батч 4, сиды 7/8/9); lora — LoRA r16 на tmix; top — верхние 4 слоя + ln_out (те же данные,
# шаги и расписание; LR — по строгой точности на валидации val.tsv одним сидом 7, затем сиды 7/8/9; сетки как 30.10).
# Проверка: 150 статей × 3 длины (~100/~400/~1000 токенов), сид 200, 16 токенов; без промпта / промпт — 12 сэмплов, остальные — 4.
# Бинарник w1101/bin_cls/gen-run (с очисткой кэша MLX в цикле вопросов и учителя; гейт gate/gate3.sh).
W=experiments/1101_dialogues; C=experiments/1101_classification; G=${GEN_RUN:-gen-run}; M=models/g1d_0.4b.safetensors
V=models/rwkv_vocab_v20230424.txt; SYS=$(cat $C/system.txt); O=$C/g1d_0p4b_bf16; mkdir -p $O
PY=python3
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
train() {  # train <плечо> <lr> <сид> <выход> <аргументы обучения…>
  local a=$1 lr=$2 s=$3 f=$4; shift 4; [ -s $f ] && return
  /usr/bin/time -l $G --model $M --vocab $V --system "$SYS" --distill --train-questions $C/train.txt --teacher-cache $C/teacher_g.json \
    --teacher-samples 4 --temp 1.0 --top-p 0.6 --max-tokens 160 --seed $s --steps 300 --lr $lr --batch 4 "$@" > ${f%.safetensors}.log 2>&1
  local rc=$?; [ -s $f ] || { echo "ОШИБКА обучения $a lr $lr s$s (rc $rc): $(tail -2 ${f%.safetensors}.log | tr '\n' ' ')"; exit 3; }
  echo "  $a lr $lr s$s rc=$rc | $(grep -E '^в обучение' ${f%.safetensors}.log | cut -c1-80) | $(grep -oE 'шаг 300: лосс [0-9.]*' ${f%.safetensors}.log | tail -1) | пик $(grep 'peak memory footprint' ${f%.safetensors}.log | awk '{printf "%.2f ГБ", $1/1e9}') | своп $(sw) | $(date '+%H:%M')"
}
gen() {  # gen <выход> <вопросы> <сэмплов> <аргументы…>
  local o=$1 q=$2 n=$3; shift 3; [ -s $o ] && return
  /usr/bin/time -l $G --model $M --vocab $V --questions $q --system "$SYS" "$@" --max-tokens 16 --samples $n --temp 1.0 --top-p 0.6 --seed 200 \
    --out $o > ${o%.json}.log 2>&1; echo "  $(basename $o) rc=$? пик $(grep 'peak memory footprint' ${o%.json}.log | awk '{printf "%.2f ГБ", $1/1e9}') $(date '+%H:%M')"
}
win() { case $1 in top) echo --top-in;; lora) echo --lora-in;; esac; }
echo "== классификация старт $(date '+%F %T') своп $(sw); ${GEN_RUN:-gen-run}"
for s in 7 8 9; do train cg 3e-4 $s $O/cg_s$s.safetensors --init fold --state-out $O/cg_s$s.safetensors; done
for spec in "lora 1e-4 3e-4 1e-3" "top 3e-5 1e-4 3e-4"; do
  set -- $spec; k=$1; shift; best=""; bestv=-1
  for lr in "$@"; do
    f=$O/${k}_lr${lr}_s7.safetensors; train $k $lr 7 $f --train $k --weights-out $f
    gen $O/val_${k}_lr${lr}_s7.json $C/val_q.txt 4 --mode none $(win $k) $f
    v=$($PY $C/vscore.py $O/val_${k}_lr${lr}_s7.json); echo "  валидация $k lr $lr: $v"
    if $PY -c "import sys; sys.exit(0 if float('$v') > float('$bestv') else 1)"; then best=$lr; bestv=$v; fi
  done
  echo "### $k: выбран lr $best (валидация $bestv)"; echo $best > $O/${k}_best_lr.txt
  for s in 8 9; do f=$O/${k}_lr${best}_s$s.safetensors; train $k $best $s $f --train $k --weights-out $f; done
done
gen $O/val_cg_s7.json $C/val_q.txt 4 --mode none --state-in $O/cg_s7.safetensors; echo "  валидация cg (lr 3e-4, протокол): $($PY $C/vscore.py $O/val_cg_s7.json)"
for k in short mid long; do
  gen $O/${k}_none.json $C/test_${k}_q.txt 12 --mode none
  gen $O/${k}_inline.json $C/test_${k}_q.txt 12 --mode inline
  for s in 7 8 9; do gen $O/${k}_cg_s$s.json $C/test_${k}_q.txt 4 --mode none --state-in $O/cg_s$s.safetensors; done
  for a in lora top; do b=$(cat $O/${a}_best_lr.txt); for s in 7 8 9; do
    gen $O/${k}_${a}_s$s.json $C/test_${k}_q.txt 4 --mode none $(win $a) $O/${a}_lr${b}_s$s.safetensors; done; done
done
echo "== классификация конец $(date '+%F %T') своп $(sw)"
