#!/bin/bash
# 01.11: гейт правки «GPU.clearCache() после каждого ответа учителя (--distill) и каждого вопроса (--questions)».
# (1) gate.sh: старые плечи матрицы побитно, ход 0 диалога = одиночный вопрос, свойства CLI;
# (2) кэш учителя: старый (w1101/bin, a112626) и новый бинарник на 20 статьях Lenta (2 сэмпла) — побитно равны, пики;
# (3) --questions на 30 длинных статьях (~1000 токенов, промпт, 2 сэмпла): ответы побитно равны, пики старый/новый.
W=experiments/1101_dialogues; C=experiments/1101_classification; T=$W/gate/g3; mkdir -p $T
M=models/g1d_0.4b.safetensors; V=models/rwkv_vocab_v20230424.txt; SYS=$(cat $C/system.txt)
OLD=${GEN_RUN:-gen-run}; NEW=${SWIFTRWKV:-../SwiftRWKV}/.build/release/gen-run
bash $W/gate/gate.sh > $T/gate1.log 2>&1; echo "(1) gate.sh rc=$?"; grep -E 'РАВНО|РАЗНЫЕ|ход 0|ГЕЙТ|НАРУШЕНО' $T/gate1.log
head -21 $C/train.txt > $T/train20.txt
for a in old new; do G=$([ $a = old ] && echo $OLD || echo $NEW); rm -f $T/teacher_$a.json
  /usr/bin/time -l $G --model $M --vocab $V --system "$SYS" --distill --train-questions $T/train20.txt --teacher-cache $T/teacher_$a.json \
    --teacher-samples 2 --temp 1.0 --top-p 0.6 --max-tokens 160 --seed 7 --init fold --steps 1 --state-out $T/s_$a.safetensors > $T/teacher_$a.log 2>&1
  echo "(2) учитель $a: пик $(grep 'peak memory footprint' $T/teacher_$a.log | awk '{printf "%.2f ГБ", $1/1e9}')"; done
cmp -s $T/teacher_old.json $T/teacher_new.json && echo "(2) кэш учителя ПОБИТНО РАВЕН" || echo "(2) кэш учителя РАЗНЫЙ"
head -30 $C/test_long_q.txt > $T/long30.txt
for a in old new; do G=$([ $a = old ] && echo $OLD || echo $NEW)
  /usr/bin/time -l $G --model $M --vocab $V --questions $T/long30.txt --system "$SYS" --mode inline --max-tokens 16 --samples 2 --temp 1.0 --top-p 0.6 --seed 200 --out $T/q_$a.json > $T/q_$a.log 2>&1
  echo "(3) вопросы $a: пик $(grep 'peak memory footprint' $T/q_$a.log | awk '{printf "%.2f ГБ", $1/1e9}')"; done
/usr/bin/python3 -c "
import json; a=json.load(open('$T/q_old.json'))['rows']; b=json.load(open('$T/q_new.json'))['rows']; print('(3) ответы', 'ПОБИТНО РАВНЫ' if a==b else 'РАЗНЫЕ')"
