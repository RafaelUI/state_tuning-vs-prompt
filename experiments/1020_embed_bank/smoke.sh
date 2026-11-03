#!/bin/bash
W=$HOME/Develop/swr_bench_keep/w1020; D=$HOME/Develop/swr_bench_keep/w1014
/usr/bin/time -l $W/bin2/embed-run --model models/files_0110/0p1b_reduction.rwkvq \
  --vocab $D/rwkv_vocab_v20230424.txt --data $HOME/Develop/swr_bench_keep/w1018/tool/tool_all.jsonl \
  --mode top:4 --task retrieval --train 2000 --eval 60 --steps 5 --batch 4 --max-tokens 2816 --max-chars 10500 \
  --seed 0 --log-every 1 --guard 11 --dedupe-pool --then-bank 5 --then-state-lr 1e-4 \
  --out $W/bank/smoke.json > $W/bank/smoke.log 2>&1
echo "rc=$?"
