#!/usr/bin/env python3
"""Сводка run_ttft.sh: на запрос = (всего − загрузка) / 100, медиана трёх кругов. Запуск из корня репозитория."""
import glob, re, statistics
W = "experiments/1103_cost"
print("| модель | строка | в строке, мс | свёрнута, мс | ускорение |\n|---|---|---|---|---|")
for m in ("0p4b", "1p5b", "2p9b"):
    for sy in ("short", "long"):
        v = {}
        for a in ("inline", "fold"):
            xs = []
            for p in sorted(glob.glob(f"{W}/{m}_{sy}_{a}_r*.log")):
                s = open(p, encoding="utf-8").read()
                tot = float(re.search(r"всего ([0-9.]+) с", s).group(1)); ld = float(re.search(r"загрузка ([0-9.]+) с", s).group(1))
                xs.append((tot - ld) / 100 * 1000)
            v[a] = statistics.median(xs)
        print(f"| {m} | {sy} | {v['inline']:.0f} | {v['fold']:.0f} | ×{v['inline'] / v['fold']:.1f} |")
