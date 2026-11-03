#!/bin/bash
# top.sh <сид> — СИЛЬНЫЙ контроль банка (21.10): первая стадия как в bank.sh (0.1B REDUCTION, верхние 4 слоя,
# 200 шагов, инструкция в строке), затем ещё 200 шагов того же обучаемого множества (верхние 4 слоя + голова) с
# инструкцией в строке (--then-top 200; шаг и расписание первой стадии; сид порядка — сид + 1, те же батчи, что у банка).
W=$HOME/Develop/swr_bench_keep/w1020; D=$HOME/Develop/swr_bench_keep/w1014
seed=$1; tag=tool_bank_top_s$seed
[ -s $W/bank/$tag.json ] && { echo "$tag | уже есть"; exit 0; }
sw0=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/')
/usr/bin/time -l $W/bin3/embed-run --model models/files_0110/0p1b_reduction.rwkvq \
  --vocab $D/rwkv_vocab_v20230424.txt --data $HOME/Develop/swr_bench_keep/w1018/tool/tool_all.jsonl \
  --mode top:4 --task retrieval --train 2000 --eval 300 --steps 200 --batch 4 --max-tokens 2816 --max-chars 10500 \
  --seed $seed --log-every 50 --guard 11 --dedupe-pool --then-top 200 \
  --out $W/bank/$tag.json > $W/bank/$tag.log 2>&1
rc=$?
sw1=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/')
echo "$tag | rc=$rc | своп за прогон $(echo "$sw1 - $sw0" | bc) МБ | footprint $(awk '/peak memory footprint/ {printf "%.2f", $1/1e9}' $W/bank/$tag.log) ГБ | $(date '+%d.%m %H:%M')"
grep -E "^ВТОРАЯ|^с инструкцией|^вторая стадия|^ещё" $W/bank/$tag.log
