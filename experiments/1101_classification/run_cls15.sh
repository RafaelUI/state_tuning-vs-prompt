#!/bin/bash
# 02.11: классификация новостей на g1d 1.5B bf16 — те же методы, что на 0.4B (решение Алексея 02.11): промпт, дистилляция, метки.
# Проба (probe15/): 1.5B со строкой на 30 статьях ~400 токенов — 8 из 60 строго верно, формат 33 из 60 → учитель есть, слабый.
# Плечи: без промпта (4 сэмпла); промпт (8); cd — состояние, дистилляция: учитель — модель со строкой, 8 сэмплов на статью,
#   16 токенов, берётся только ответ «ровно верная рубрика» (кэш teacher15.json); cg — состояние на золотых метках (teacher_g.json).
# Обучение — протокол c4 (fold, 300 шагов, lr 3e-4, батч 4, отсечение 1.0), сиды 7/8/9; состояния — 4 сэмпла.
# Проверка — те же 150 статей × 3 длины, сид 200, 16 токенов. Бинарник w1101/bin_cls/gen-run (c332453).
W=experiments/1101_dialogues; C=experiments/1101_classification; G=${GEN_RUN:-gen-run}; M=models/g1d_1.5b.safetensors
V=models/rwkv_vocab_v20230424.txt; SYS=$(cat $C/system.txt); O=$C/g1d_1p5b_bf16; mkdir -p $O
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
train() {  # train <плечо> <кэш> <сид> <сэмплов учителя>
  local a=$1 c=$2 s=$3 ts=$4; local f=$O/${a}_s$s.safetensors; [ -s $f ] && return
  /usr/bin/time -l $G --model $M --vocab $V --system "$SYS" --distill --train-questions $C/train.txt --teacher-cache $c \
    --teacher-samples $ts --temp 1.0 --top-p 0.6 --max-tokens 16 --reject-regex "System:|классификатор" \
    --seed $s --init fold --steps 300 --lr 3e-4 --batch 4 --state-out $f > $O/distill_${a}_s$s.log 2>&1
  local rc=$?; [ -s $f ] || { echo "ОШИБКА обучения $a s$s (rc $rc): $(grep -E '^учитель|^в обучение|нечему' $O/distill_${a}_s$s.log | tr '\n' ' ')"; return 1; }
  echo "  $a s$s rc=$rc | $(grep -E '^учитель \(|^в обучение' $O/distill_${a}_s$s.log | tr '\n' ' ' | cut -c1-180) | $(grep -oE 'шаг 300: лосс [0-9.]*' $O/distill_${a}_s$s.log) | пик $(grep 'peak memory footprint' $O/distill_${a}_s$s.log | awk '{printf "%.2f ГБ", $1/1e9}') | своп $(sw) | $(date '+%H:%M')"
}
gen() {  # gen <выход> <вопросы> <сэмплов> <аргументы…>
  local o=$1 q=$2 n=$3; shift 3; [ -s $o ] && return
  /usr/bin/time -l $G --model $M --vocab $V --questions $q --system "$SYS" "$@" --max-tokens 16 --samples $n --temp 1.0 --top-p 0.6 --seed 200 \
    --out $o > ${o%.json}.log 2>&1; echo "  $(basename $o) rc=$? пик $(grep 'peak memory footprint' ${o%.json}.log | awk '{printf "%.2f ГБ", $1/1e9}') $(date '+%H:%M')"
}
echo "== классификация 1.5B старт $(date '+%F %T') своп $(sw); ${GEN_RUN:-gen-run}"
for s in 7 8 9; do train cd $C/teacher15.json $s 8 || break; done        # учитель считается при первом вызове
for s in 7 8 9; do train cg $C/teacher_g.json $s 8; done
for k in short mid long; do
  gen $O/${k}_none.json $C/test_${k}_q.txt 4 --mode none
  gen $O/${k}_inline.json $C/test_${k}_q.txt 8 --mode inline
  for a in cd cg; do for s in 7 8 9; do [ -s $O/${a}_s$s.safetensors ] && gen $O/${k}_${a}_s$s.json $C/test_${k}_q.txt 4 --mode none --state-in $O/${a}_s$s.safetensors; done; done
done
echo "== классификация 1.5B конец $(date '+%F %T') своп $(sw)"
