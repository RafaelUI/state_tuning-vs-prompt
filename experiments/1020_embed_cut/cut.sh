#!/bin/bash
# fold.sh <сид> — то же сравнение, но обрезка 512 токенов в оценке (20.10), только качество.
# Первая стадия как в серии w1018 (qt2.sh: 0.1B REDUCTION, верхние 4 слоя, 200 шагов,
# max-tokens 2816, max-chars 10500, слияние кандидатов), затем --compare-folded:
# оба пути оценки на одних весах, 3 круга вперемежку (первый — прогрев).
W=$HOME/Develop/swr_bench_keep/w1020; D=$HOME/Develop/swr_bench_keep/w1014
seed=$1; tag=tool_cut512_s$seed
[ -s $W/cut/$tag.json ] && { echo "$tag | уже есть"; exit 0; }
sw0=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/')
/usr/bin/time -l $W/bin/embed-run --model models/files_0110/0p1b_reduction.rwkvq \
  --vocab $D/rwkv_vocab_v20230424.txt --data $HOME/Develop/swr_bench_keep/w1018/tool/tool_all.jsonl \
  --mode top:4 --task retrieval --train 2000 --eval 300 --steps 200 --batch 4 --max-tokens 2816 --max-chars 10500 \
  --seed $seed --log-every 50 --guard 11 --dedupe-pool --compare-folded --compare-rounds 1 --compare-max-tokens 512 \
  --out $W/cut/$tag.json > $W/cut/$tag.log 2>&1
rc=$?
sw1=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/')
echo "$tag | rc=$rc | своп за прогон $(echo "$sw1 - $sw0" | bc) МБ | footprint $(awk '/peak memory footprint/ {printf "%.2f", $1/1e9}' $W/cut/$tag.log) ГБ | $(date '+%d.%m %H:%M')"
grep -E "^СВЁРТКА|ВНИМАНИЕ|после обучения" $W/cut/$tag.log
