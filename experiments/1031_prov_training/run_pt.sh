#!/bin/bash
# 31.10, этап 2: провокации в обучении. Протокол c4 (fold, 300 шагов, lr 3e-4, батч 4, сиды 7/8/9), бинарник w1027/bin/gen-run (c689590) —
# тот же, что у матрицы (w1027/m) и провокаций (w1030/prov): старые плечи c4 / без промпта / промпт сравниваются без пересчёта.
# Плечи (вложенный план):
#   c4   — матрица (w1027/m/<модель>), как есть;
#   c4c  — только 1.5B: те же данные c4 без ответов вида id, присваивающих чужую личность (у 0.4B таких 0 → c4c ≡ c4);
#   c4pd — c4c + 32 новых вопроса (28 наводящих «нет, я Neuron» + 4 «ты Neuron? — да»), ответы учителя: 16 сэмплов, строгий
#          регэксп + ручная сверка (кэш pt/<модель>/teacher_d.json) — самодистилляция;
#   c4ph — c4c + те же 32 вопроса с ручными ответами hand_answers.tsv (кэш teacher_h.json) — дообучение на данных.
#   (решение Алексея 31.10: оба плеча — учитель 0.4B на новых вопросах почти не даёт нужного ответа.)
#   c4c − c4 = очистка, c4pd/c4ph − c4c = провокации в обучении.
# Проверки у каждого состояния: общие (checks_q s4, сид 200, 80), имя (questions s10, сид 100, 120), провокации
# (w1030/provocations.txt s10, сид 300, 120), перенос (transfer_q s4, сид 200, 80), «не отрицает ли своё» (deny_q s10, сид 400, 120);
# deny_q ещё у без промпта / промпта / c4 × 3 (нового вопросника у них нет).
W=experiments/1031_prov_training; N=data/neuron; B=experiments/1027_matrix
V=models/rwkv_vocab_v20230424.txt; G=${GEN_RUN:-gen-run}; SYS=$(cat $N/system.txt); PR=experiments/1030_provocations_control/provocations.txt
path() { case $1 in g1d_0p4b_bf16) echo models/g1d_0.4b.safetensors;; g1d_1p5b_bf16) echo $B/g1d_1.5b.safetensors;; esac; }
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
train() {  # train <модель> <плечо> <сид> <файл вопросов> <кэш учителя>
  local m=$1 a=$2 s=$3 tq=$4 tc=$5 O=$W/pt/$1; local f=$O/${a}_s$s.safetensors
  [ -s $f ] && return
  /usr/bin/time -l $G --model $(path $m) --vocab $V --system "$SYS" --distill --train-questions $tq \
    --teacher-cache $tc --teacher-samples 4 --temp 1.0 --top-p 0.6 --max-tokens 160 \
    --reject-regex "Ты — Neuron|System:" --seed $s --init fold --steps 300 --lr 3e-4 --batch 4 \
    --state-out $f > $O/distill_${a}_s$s.log 2>&1
  local rc=$?
  [ -s $f ] || { echo "ОШИБКА обучения $m $a s$s (rc $rc): $(tail -3 $O/distill_${a}_s$s.log | tr '\n' ' ')"; exit 3; }
  echo "  $m $a s$s rc=$rc | $(grep -E '^учитель \(|^в обучение' $O/distill_${a}_s$s.log | tr '\n' ' ' | cut -c1-200) | $(grep -o 'шаг 300: лосс [0-9.]*' $O/distill_${a}_s$s.log) | пик $(grep 'peak memory footprint' $O/distill_${a}_s$s.log | awk '{printf "%.2f ГБ", $1/1e9}') | своп $(sw) | $(date '+%H:%M')"
}
gen() {  # gen <модель> <выход> <вопросы> <сэмплов> <токенов> <сид> <аргументы…>
  local m=$1 o=$2 q=$3 n=$4 k=$5 sd=$6; shift 6; [ -s $o ] && return
  $G --model $(path $m) --vocab $V --questions $q --system "$SYS" "$@" --max-tokens $k --samples $n \
    --temp 1.0 --top-p 0.6 --seed $sd --out $o > ${o%.json}.log 2>&1; echo "  $(basename $(dirname $o))/$(basename $o) rc=$? $(date '+%H:%M')"
}
checks() {  # checks <модель> <плечо> <сид>
  local m=$1 a=$2 s=$3 O=$W/pt/$1; local st=$O/${a}_s$s.safetensors
  gen $m $O/v2_${a}_s${s}_s4.json $N/checks_q.txt 4 80 200 --mode none --state-in $st
  gen $m $O/id_${a}_s${s}_s10.json $N/questions.txt 10 120 100 --mode none --state-in $st
  gen $m $O/prov_${a}_s$s.json $PR 10 120 300 --mode none --state-in $st
  gen $m $O/v3_${a}_s${s}_s4.json $N/transfer_q.txt 4 80 200 --mode none --state-in $st
  gen $m $O/deny_${a}_s$s.json $W/deny_q.txt 10 120 400 --mode none --state-in $st
}
STAGE=${STAGE:-"c4c teacher16 c4ph base"}   # этапы по порядку; c4pd — после ручной сверки teacher_d.json
echo "== провокации в обучении [$STAGE] старт $(date '+%F %T') своп $(sw); ${GEN_RUN:-gen-run}"
for st in $STAGE; do case $st in
  c4c) m=g1d_1p5b_bf16; echo "### $m c4c (очистка id) $(date '+%H:%M')"
       for s in 7 8 9; do train $m c4c $s $N/train_questions4.txt $W/pt/$m/teacher.json; checks $m c4c $s; done;;
  teacher16) for m in g1d_0p4b_bf16 g1d_1p5b_bf16; do O=$W/pt/$m   # догенерировать 32 новых вопроса × 16 сэмплов (сиды 7..22)
       /usr/bin/time -l $G --model $(path $m) --vocab $V --system "$SYS" --distill --train-questions $W/train_prov.txt \
         --teacher-cache $O/teacher_d.json --teacher-samples 16 --temp 1.0 --top-p 0.6 --max-tokens 160 \
         --reject-regex "Ты — Neuron|System:" --seed 7 --init fold --steps 1 --lr 3e-4 --batch 4 \
         --state-out $O/probe16_1step.safetensors > $O/teacher16.log 2>&1
       echo "  $m учитель16 rc=$? | $(grep -E '^учитель \(|^в обучение' $O/teacher16.log | tr '\n' ' ') | $(date '+%H:%M')"; done;;
  c4ph|c4pd) c=$([ $st = c4ph ] && echo teacher_h.json || echo teacher_d.json)
       for m in g1d_0p4b_bf16 g1d_1p5b_bf16; do echo "### $m $st $(date '+%H:%M')"
         for s in 7 8 9; do train $m $st $s $W/train_prov.txt $W/pt/$m/$c; checks $m $st $s; done; done;;
  base) for m in g1d_0p4b_bf16 g1d_1p5b_bf16; do O=$W/pt/$m; echo "### $m опорные плечи на deny_q $(date '+%H:%M')"
       gen $m $O/deny_none.json $W/deny_q.txt 10 120 400 --mode none
       gen $m $O/deny_inline.json $W/deny_q.txt 10 120 400 --mode inline
       for s in 7 8 9; do gen $m $O/deny_c4_s$s.json $W/deny_q.txt 10 120 400 --mode none --state-in $B/m/$m/c4_s$s.safetensors; done; done;;
esac; done
echo "== провокации в обучении [$STAGE] конец $(date '+%F %T') своп $(sw)"
