#!/usr/bin/env python3
"""Сводка «свёрнутая инструкция против инструкции в строке» (w1019, Tool_Use).

На каждом сиде — одни веса после первой стадии и две оценки на одних и тех же
300 отложенных запросах: инструкция в строке и свёрнутая. Δ — среднее по сидам
поштучных разностей 1/ранг, 95 %-интервал — парный бутстрэп по запросам.
Сверка: плечо «в строке» обязано совпасть с «с инструкцией» серии w1018 (тот же
сид, та же первая стадия).

    python3 foldsum.py [--boot 10000]
"""
import glob, json, os, re, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
BOOT = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 10000
rng = np.random.default_rng(0)
R = {}
for p in sorted(glob.glob(W + "/tool_fold_s*.json")):
    seed = int(re.search(r"_s(\d+)\.json$", p).group(1))
    R[seed] = json.load(open(p))
if not R:
    sys.exit("нет прогонов")
inv = lambda r, k: 1.0 / np.array(r[k], dtype=float)
med = lambda a: float(np.median(a))

print("| сид | в строке | свёрнутая | разошлось строк | время в строке, с | свёрнутая, с | ускорение | пик MLX, ГБ |")
print("|---|---|---|---|---|---|---|---|")
for s, r in sorted(R.items()):
    ti, tf = med(r["cmp_inline_seconds"]), med(r["cmp_folded_seconds"])
    print(f"| {s} | {r['cmp_inline']:.4f} | {r['cmp_folded']:.4f} | {r['cmp_rows_differ']} "
          f"(необрезанных {r['cmp_rows_differ_uncut']}) | {ti:.1f} | {tf:.1f} | {ti / tf:.2f}x "
          f"| {med(r['cmp_inline_peak_gb']):.2f} / {med(r['cmp_folded_peak_gb']):.2f} |")

d = np.mean([inv(r, "per_row_cmp_folded") - inv(r, "per_row_cmp_inline") for r in R.values()], axis=0)
boot = d[rng.integers(0, d.size, size=(BOOT, d.size))].mean(axis=1)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\n- свёрнутая − в строке: {d.mean():+.1e} [{lo:+.1e}; {hi:+.1e}] ({len(R)} с.); "
      f"строк с ненулевой разностью (среднее по сидам): {(d != 0).sum()}")
r0 = next(iter(R.values()))
n = r0["eval"] if "eval" in r0 else len(r0["per_row_cmp_inline"])
print(f"- запросов {n}, с инструкцией {r0['cmp_rows_with_instruction']}, различных инструкций "
      f"{r0['cmp_distinct_instructions']}; токенов запросов в строке {r0['cmp_inline_tokens']} "
      f"({r0['cmp_inline_tokens'] / n:.0f} на запрос), хвостов {r0['cmp_tail_tokens']} "
      f"({r0['cmp_tail_tokens'] / n:.0f} на запрос) + свёртка {r0['cmp_fold_tokens']} один раз; "
      f"всего токенов {r0['cmp_inline_tokens']} против {r0['cmp_tail_tokens'] + r0['cmp_fold_tokens']} "
      f"({r0['cmp_inline_tokens'] / (r0['cmp_tail_tokens'] + r0['cmp_fold_tokens']):.1f}x); "
      f"обрезано в строке {r0['cmp_inline_cut']}, хвостов {r0['cmp_tail_cut']}")
same = []
for s, r in R.items():
    p = os.path.expanduser(f"~/Develop/swr_bench_keep/w1018/q/tool_q_all_s{s}.json")
    if os.path.exists(p):
        same.append((s, json.load(open(p))["per_row_instruct"] == r["per_row_cmp_inline"]))
print(f"- плечо «в строке» = «с инструкцией» серии w1018 (поштучно): {same if same else '—'}")
