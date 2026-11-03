#!/bin/bash
# ждать конца замера обрезки и полного прогона (com.swr.cut, com.swr.full9), затем 3 сида × (банк, контроль)
W=$HOME/Develop/swr_bench_keep/w1020
for l in cut full9; do while ! grep -q "^EXIT" $HOME/Develop/swr_bench_keep/w1014/bg_$l.log 2>/dev/null; do sleep 30; done; done
for s in 0 1 2; do for arm in bank head; do $W/bank/bank.sh $s $arm; done; done
