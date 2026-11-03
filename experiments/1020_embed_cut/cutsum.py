#!/usr/bin/env python3
"""Сводка «в строке против свёрнутой при обрезке 512 токенов» (w1020, Tool_Use).

Веса — первая стадия при --max-tokens 2816 (те же, что в w1019); оценка обоих
путей при 512. Δ — среднее по сидам поштучных разностей 1/ранг, 95 % — парный
бутстрэп; отдельно по запросам, которые в строке обрезаны (запрос вытеснен
инструкцией), и по необрезанным.

    python3 cutsum.py
"""
import glob, json, os, re
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
R = {int(re.search(r"_s(\d+)\.json$", p).group(1)): json.load(open(p))
     for p in glob.glob(W + "/tool_cut512_s*.json")}
inv = lambda r, k: 1.0 / np.array(r[k], dtype=float)


def boot(d):
    b = d[rng.integers(0, d.size, size=(10000, d.size))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return f"{d.mean():+.4f} [{lo:+.4f}; {hi:+.4f}] (n={d.size})"


print("| сид | в строке (512) | свёрнутая (512) | обрезано в строке | MRR обрезанных: в строке / свёрнутая |")
print("|---|---|---|---|---|")
for s, r in sorted(R.items()):
    cut = np.zeros(len(r["per_row_cmp_inline"]), bool); cut[r["cmp_cut_rows"]] = True
    a, b = inv(r, "per_row_cmp_inline"), inv(r, "per_row_cmp_folded")
    print(f"| {s} | {r['cmp_inline']:.4f} | {r['cmp_folded']:.4f} | {cut.sum()} | {a[cut].mean():.4f} / {b[cut].mean():.4f} |")
d = np.mean([inv(r, "per_row_cmp_folded") - inv(r, "per_row_cmp_inline") for r in R.values()], axis=0)
print(f"\n- свёрнутая − в строке: {boot(d)} ({len(R)} с.)")
cuts = [set(r["cmp_cut_rows"]) for r in R.values()]
print(f"- обрезанные строки одни и те же на всех сидах: {all(c == cuts[0] for c in cuts)}")
m = np.zeros(d.size, bool); m[list(cuts[0])] = True
print(f"- на обрезанных: {boot(d[m])}; на необрезанных: {boot(d[~m])}")
f19 = {s: json.load(open(os.path.expanduser(f"~/Develop/swr_bench_keep/w1019/fold/tool_fold_s{s}.json")))
       for s in R if os.path.exists(os.path.expanduser(f"~/Develop/swr_bench_keep/w1019/fold/tool_fold_s{s}.json"))}
print(f"- свёрнутая при 512 = свёрнутой при 2816 (w1019, поштучно): "
      f"{[R[s]['per_row_cmp_folded'] == f19[s]['per_row_cmp_folded'] for s in sorted(f19)]}")
