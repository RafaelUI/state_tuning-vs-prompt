#!/usr/bin/env python3
"""Сводка «своё состояние на каждую инструкцию» (w1020, Tool_Use).

На каждом сиде одна первая стадия (инструкция в строке), затем две вторые:
bank — обучается банк со старта свёрнутых инструкций; head — контроль, те же
шаги обучают голову при свёрнутом банке. Оценки на одних и тех же 300
отложенных запросах. Δ — среднее по сидам поштучных разностей 1/ранг,
95 %-интервал — парный бутстрэп по запросам.

    python3 banksum.py [--boot 10000]
"""
import glob, json, os, re, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
BOOT = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else 10000
rng = np.random.default_rng(0)
R = {}
for p in glob.glob(W + "/tool_bank_*_s?.json"):
    arm, seed = re.search(r"tool_bank_(bank|head|top)_s(\d+)\.json$", p).groups()
    R[(arm, int(seed))] = json.load(open(p))
if not R:
    sys.exit("нет прогонов")
inv = lambda r, k: 1.0 / np.array(r[k], dtype=float)


def paired(xs, ys, mask=None):
    seeds = sorted(set(xs) & set(ys))
    if not seeds:
        return "—"
    d = np.mean([xs[s] - ys[s] for s in seeds], axis=0)
    if mask is not None:
        d = d[mask]
    if d.size == 0:
        return "—"
    boot = d[rng.integers(0, d.size, size=(BOOT, d.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f"{d.mean():+.4f} [{lo:+.4f}; {hi:+.4f}] ({len(seeds)} с., n={d.size})"


cell = lambda arm, key: {s: inv(r, key) for (a, s), r in R.items() if a == arm and key in r}
mean = lambda c: f"{np.mean([v.mean() for v in c.values()]):.4f}" if c else "—"
print("| плечо | сидов | с инструкцией | банк свёрнутых | после второй стадии |")
print("|---|---|---|---|---|")
for arm in ("bank", "head", "top"):
    print(f"| {arm} | {len(cell(arm, 'per_row'))} | {mean(cell(arm, 'per_row_instruct'))} "
          f"| {mean(cell(arm, 'per_row_bank_folded'))} | {mean(cell(arm, 'per_row'))} |")
b, h = cell("bank", "per_row"), cell("head", "per_row")
ins, fold = cell("bank", "per_row_instruct"), cell("bank", "per_row_bank_folded")
print()
print(f"- банк свёрнутых − с инструкцией: {paired(fold, ins)}")
print(f"- банк обучен − с инструкцией: {paired(b, ins)}")
print(f"- контроль (голова) − с инструкцией: {paired(h, ins)}")
print(f"- банк обучен − контроль: {paired(b, h)}")
tp = cell("top", "per_row")
if tp:
    print(f"- СИЛЬНЫЙ контроль (ещё 200 шагов верхних слоёв с инструкцией) − с инструкцией: {paired(tp, ins)}")
    print(f"- банк обучен − сильный контроль: {paired(b, tp)}")
r0 = R.get(("bank", 0)) or next(iter(R.values()))
unseen = np.array(r0.get("per_row_unseen", []), dtype=bool)
if unseen.size and unseen.any():
    print(f"- на запросах с инструкцией, которой нет в обучении: банк − с инструкцией {paired(b, ins, unseen)}")
    print(f"- на остальных: банк − с инструкцией {paired(b, ins, ~unseen)}; банк − контроль {paired(b, h, ~unseen)}")
st = [R[("bank", s)]["per_row_instruct"] == R[("head", s)]["per_row_instruct"]
      for s in range(3) for o in ("head", "top") if ("bank", s) in R and (o, s) in R]
print(f"- первая стадия одинакова у плеч одного сида: {all(st) if st else '—'} ({len(st)} пар)")
w = os.path.expanduser("~/Develop/swr_bench_keep/w1018/q")
same = [json.load(open(f"{w}/tool_q_all_s{s}.json"))["per_row_instruct"] == r["per_row_instruct"]
        for (a, s), r in R.items() if os.path.exists(f"{w}/tool_q_all_s{s}.json")]
print(f"- первая стадия = серии w1018 (поштучно): {all(same) if same else '—'} ({len(same)})")
print(f"- банк: {r0.get('bank_size')} инструкций, {r0.get('bank_mb', 0):.0f} МБ; в отложенных без обучения: "
      f"{r0.get('bank_unseen_instructions')} инструкций, {r0.get('bank_unseen_rows')} запросов")
for (a, s), r in sorted(R.items()):
    print(f"  {a} s{s}: вторая стадия {r.get('second_stage_seconds', 0):.0f} с, лосс {r.get('state_final_loss', 0):.4f}")
