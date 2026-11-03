#!/usr/bin/env python3
"""Draws the figures for README.md / README.ru.md and docs/.

All numbers are copied from the tables in docs/results.md (the section is named
in a comment above each figure); nothing is recomputed here. To change a number,
fix it in docs/results.md and in the DATA block below, then run

    python3 figures/make_figures.py

which writes figures/en/*.png and figures/ru/*.png. Needs matplotlib.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- palette (colour-blind safe, same meaning in every figure) -------------
C_NONE = "#8a8f98"    # no prompt
C_PROMPT = "#d9822b"  # prompt as text
C_STATE = "#2a6fdb"   # trained state
C_STATE_LIGHT = "#9dbcf0"
C_LORA = "#7b4fb3"
C_TOP4 = "#2a9d6f"
INK = "#222222"
GRID = "#e3e5e8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#555555",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.dpi": 100,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# ---- strings: one dict per language -----------------------------------------
S = {
    "en": {
        "no_prompt": "no prompt", "prompt": "prompt as text", "state": "trained state",
        "delta_acc": "Δ accuracy vs the no-prompt model",
        # overview
        "ov_prompt": "System prompt\n(text, re-read on\nevery request)",
        "ov_fold": "Fold once:\nrun the model over\nthe prompt",
        "ov_state": "State\n(fixed size,\nreusable)",
        "ov_req": "Each request:\nonly the question's\ntokens",
        "ov_teacher": "Teacher = model\nwith the prompt;\nkeep only good answers",
        "ov_train": "Train the state\n(weights frozen,\n300 steps)",
        "ov_trained": "Trained state\n(1.6–5.4 M params,\n6.5–22 MB)",
        "ov_same": "same behaviour,\ncheaper",
        "ov_start": "training starts\nfrom the fold",
        "ov_maybe": "can be better\nthan the prompt",
        "ov_title": "Two ways to replace a system prompt with an initial RWKV state",
        # matrix
        "mx_title": "Prompt vs trained state on eight base models",
        "mx_sub": "199 questions with a checkable answer; points with 95 % intervals; hollow marker = state's difference is not significant",
        # c3/c4
        "c_title": "What the state learns depends on what the teacher got right",
        "c_cats": ["all (199)", "translation (32)", "arithmetic (36)", "other facts (131)"],
        "c_c3": "state, 31 verified facts in training",
        "c_c4": "state, 109 verified facts in training",
        "c_note": "g1d 0.4B, 3 training seeds; whiskers: 95 % interval",
        # transfer
        "t_title": "Does the learned answer style survive a new wording?",
        "t_cats": ["training template", "new wording"],
        "t_sub04": "0.4B", "t_sub15": "1.5B",
        # five prompts
        "p_title": "Six different prompts, two model sizes",
        "p_names": ["Neuron (writes poems)", "Orion (a name only)", "plain (no markdown)",
                    "short (one sentence)", "English only", "JSON only"],
        "p_note": "JSON on 0.4B: the teacher gave only 58 correct facts",
        "p_x": "Δ accuracy vs the no-prompt model",
        # provocations
        "v_title": "Share of answers where the model claims someone else's identity",
        "v_sub": "20 leading / open questions (“Are you Claude from Anthropic?”), lower is better",
        "v_arms": ["no prompt", "prompt", "state c4", "state c4 cleaned", "state + self-distilled refusals", "state + hand-written refusals"],
        "v_short": ["no prompt", "prompt", "c4", "c4c", "c4pd", "c4ph"],
        # dialogues
        "d_title": "Role over a conversation, 1.5B",
        "d_role": "answers “Neuron” to “what is your name?”",
        "d_facts": "control: simple fact stays correct?",
        "d_turn": "turn of the conversation",
        "d_pc": "state c4 + hand-written refusals",
        "d_state": "state c4",
        "d_np": "no prompt",
        # classification
        "k_title": "News classification (5 classes): following an instruction a 0.4B model cannot follow from text",
        "k_a": "0.4B, by document length", "k_b": "1.5B, by document length", "k_c": "0.4B, by training set size",
        "k_acc": "strict accuracy", "k_tok": "document length, tokens", "k_n": "training articles",
        "k_lora": "LoRA r16", "k_top": "top-4 layers", "k_dist": "state, distilled from prompt",
        "k_gold": "state, gold labels",
        # cost
        "x_title": "Time to first token: prompt as text vs folded prompt",
        "x_short": "26-token prompt", "x_long": "1206-token prompt",
        "x_ms": "ms per request", "x_text": "prompt as text", "x_fold": "folded / state",
        # misc
        "bf16": "bf16",
    },
    "ru": {
        "no_prompt": "без промпта", "prompt": "промпт в строке", "state": "обученное состояние",
        "delta_acc": "Δ точности к модели без промпта",
        "ov_prompt": "Системный промпт\n(текст, обрабатывается\nв каждом запросе)",
        "ov_fold": "Свернуть один раз:\nпрогнать модель\nпо промпту",
        "ov_state": "Состояние\n(фиксированный\nразмер)",
        "ov_req": "Каждый запрос:\nтолько токены\nвопроса",
        "ov_teacher": "Учитель = модель\nс промптом;\nберём только хорошие ответы",
        "ov_train": "Обучаем состояние\n(веса заморожены,\n300 шагов)",
        "ov_trained": "Обученное состояние\n(1.6–5.4 M параметров,\n6.5–22 МБ)",
        "ov_same": "то же поведение,\nдешевле",
        "ov_start": "обучение стартует\nсо свёртки",
        "ov_maybe": "может быть лучше\nпромпта",
        "ov_title": "Два способа заменить системный промпт начальным состоянием RWKV",
        "mx_title": "Промпт и обученное состояние на восьми базах",
        "mx_sub": "199 вопросов с проверяемым ответом; точки с 95 %-ми интервалами; пустой маркер — разница состояния не значима",
        "c_title": "Чему учится состояние, зависит от того, что верно ответил учитель",
        "c_cats": ["все (199)", "перевод (32)", "арифметика (36)", "остальные факты (131)"],
        "c_c3": "состояние, 31 верный факт в обучении",
        "c_c4": "состояние, 109 верных фактов в обучении",
        "c_note": "g1d 0.4B, 3 сида обучения; усы: 95 %-й интервал",
        "t_title": "Переносится ли выученная манера ответа на новую формулировку?",
        "t_cats": ["шаблон из обучения", "новая формулировка"],
        "t_sub04": "0.4B", "t_sub15": "1.5B",
        "p_title": "Шесть разных промптов, два размера модели",
        "p_names": ["Neuron (пишет стихи)", "Orion (только имя)", "plain (без markdown)",
                    "short (одним предложением)", "только английский", "только JSON"],
        "p_note": "JSON на 0.4B: учитель дал лишь 58 верных фактов",
        "p_x": "Δ точности к модели без промпта",
        "v_title": "Доля ответов, где модель присваивает чужую личность",
        "v_sub": "20 наводящих и открытых вопросов («Ты Claude от Anthropic?»), меньше — лучше",
        "v_arms": ["без промпта", "промпт", "состояние c4", "c4 после очистки", "+ отказы самодистилляцией", "+ ручные отказы"],
        "v_short": ["без промпта", "промпт", "c4", "c4c", "c4pd", "c4ph"],
        "d_title": "Роль на протяжении диалога, 1.5B",
        "d_role": "отвечает «Neuron» на вопрос об имени",
        "d_facts": "контроль: простой факт по-прежнему верен?",
        "d_turn": "ход диалога",
        "d_pc": "состояние c4 + ручные отказы",
        "d_state": "состояние c4",
        "d_np": "без промпта",
        "k_title": "Классификация новостей (5 рубрик): инструкция, которую 0.4B не выполняет из текста",
        "k_a": "0.4B, по длине документа", "k_b": "1.5B, по длине документа", "k_c": "0.4B, по объёму обучения",
        "k_acc": "строгая точность", "k_tok": "длина документа, токенов", "k_n": "статей в обучении",
        "k_lora": "LoRA r16", "k_top": "верхние 4 слоя", "k_dist": "состояние, дистилляция из промпта",
        "k_gold": "состояние, золотые метки",
        "x_title": "Время до первого токена: промпт в строке и свёрнутый промпт",
        "x_short": "промпт 26 токенов", "x_long": "промпт 1206 токенов",
        "x_ms": "мс на запрос", "x_text": "промпт в строке", "x_fold": "свёрнут / состояние",
        "bf16": "bf16",
    },
}


def save(fig, lang, name):
    out = os.path.join(HERE, lang)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)


def label_bars(ax, bars, fmt="{:+.2f}", dy=0.012, size=8):
    for b in bars:
        h = b.get_height()
        txt = fmt.format(h)
        if abs(h) < 0.005 and "+" in fmt:
            txt = "0.00"
        ax.text(b.get_x() + b.get_width() / 2, h + (dy if h >= 0 else -dy), txt,
                ha="center", va="bottom" if h >= 0 else "top", fontsize=size)


# ---------------------------------------------------------------------------
# 01. Overview diagram
# ---------------------------------------------------------------------------
def fig_overview(lang):
    t = S[lang]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, ec, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=tc)

    def arrow(x1, y1, x2, y2, color="#555555", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                     lw=1.4, color=color))

    # row 1: folding
    box(1, 27, 20, 14, t["ov_prompt"], "#fdeede", C_PROMPT)
    box(30, 27, 20, 14, t["ov_fold"], "#f3f4f6", "#777777")
    box(59, 27, 18, 14, t["ov_state"], "#e4edfc", C_STATE)
    box(85, 27, 14, 14, t["ov_req"], "#f3f4f6", "#777777")
    arrow(21.8, 34, 29.2, 34)
    arrow(50.8, 34, 58.2, 34)
    arrow(77.8, 34, 84.2, 34)
    ax.text(71, 22.0, t["ov_same"], ha="left", va="center", fontsize=9, color=C_STATE, style="italic")

    # row 2: training
    box(1, 3, 22, 14, t["ov_teacher"], "#fdeede", C_PROMPT)
    box(33, 3, 22, 14, t["ov_train"], "#f3f4f6", "#777777")
    box(65, 3, 22, 14, t["ov_trained"], "#e4edfc", C_STATE)
    arrow(23.8, 10, 32.2, 10)
    arrow(55.8, 10, 64.2, 10)
    ax.text(76, 0.5, t["ov_maybe"], ha="center", va="top", fontsize=9, color=C_STATE, style="italic")
    arrow(68, 18, 68, 26, color=C_STATE, style="<|-")  # trained state starts from the folded prompt
    ax.text(66.5, 22.0, t["ov_start"], ha="right", va="center", fontsize=8.5, color="#555555")
    fig.suptitle(t["ov_title"], fontsize=12, y=0.98)
    save(fig, lang, "01_overview")


# ---------------------------------------------------------------------------
# 02. Eight bases (docs/results.md, "Scale × quantization")
# ---------------------------------------------------------------------------
# name, (prompt delta, lo, hi) [bootstrap over questions], (state delta, lo, hi) [two-level bootstrap seeds x questions]
# intervals are read from experiments/1027_matrix/m/*/summary.md; "significant" = the state interval excludes 0
MATRIX = [
    ("g1d 0.4B bf16", (-0.143, -0.182, -0.106), (+0.057, +0.014, +0.100)),
    ("g1d 0.4B REDUCTION", (-0.131, -0.170, -0.093), (+0.036, -0.008, +0.081)),
    ("g1d 0.4B COMPRESSION", (-0.151, -0.194, -0.110), (+0.030, -0.013, +0.071)),
    ("g1d 1.5B bf16", (-0.039, -0.060, -0.016), (+0.038, +0.013, +0.062)),
    ("g1d 1.5B REDUCTION", (-0.051, -0.072, -0.030), (+0.048, +0.023, +0.072)),
    ("g1d 1.5B COMPRESSION", (-0.042, -0.065, -0.019), (+0.013, -0.014, +0.038)),
    ("g1j 1.5B COMPRESSION", (-0.116, -0.142, -0.090), (-0.006, -0.034, +0.021)),
    ("g1j 2.9B COMPRESSION", (-0.034, -0.059, -0.008), (+0.047, +0.026, +0.070)),
]


def fig_matrix(lang):
    t = S[lang]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ys = list(range(len(MATRIX)))[::-1]
    for y, (name, (dp, pl, ph), (ds, sl, sh)) in zip(ys, MATRIX):
        sig = sl > 0 or sh < 0
        ax.plot([dp, ds], [y, y], color="#e3e5e8", lw=2, zorder=1)
        ax.plot([pl, ph], [y + 0.14, y + 0.14], color=C_PROMPT, lw=1.6, zorder=2)
        ax.plot([sl, sh], [y - 0.14, y - 0.14], color=C_STATE, lw=1.6, zorder=2)
        ax.scatter([dp], [y + 0.14], s=48, color=C_PROMPT, zorder=3)
        ax.scatter([ds], [y - 0.14], s=48, color=C_STATE if sig else "white", edgecolor=C_STATE, linewidth=1.8, zorder=3)
    ax.axvline(0, color="#555555", lw=1.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in MATRIX])
    ax.set_xlabel(t["delta_acc"])
    ax.set_xlim(-0.205, 0.11)
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=48, color=C_PROMPT, label=t["prompt"])
    ax.scatter([], [], s=48, color=C_STATE, label=t["state"])
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title(t["mx_title"], fontsize=12, loc="left", pad=22)
    ax.text(0, 1.02, t["mx_sub"], transform=ax.transAxes, fontsize=8.5, color="#555555", va="bottom")
    save(fig, lang, "02_matrix")


# ---------------------------------------------------------------------------
# 03. c3 -> c4 (docs/results.md, "Generation: Neuron", big check set)
# ---------------------------------------------------------------------------
def fig_c3c4(lang):
    t = S[lang]
    prompt = [-0.129, -0.029, +0.002, -0.190]
    c3 = [-0.043, +0.214, +0.125, -0.152]
    c4 = [+0.063, +0.380, +0.201, -0.052]
    # 95 % intervals, paired bootstrap over questions (reproduced with experiments/1020_neuron_0p4b/paired2.py)
    ci_prompt = [(-0.164, -0.095), (-0.081, +0.026), (-0.069, +0.072), (-0.234, -0.146)]
    ci_c3 = [(-0.089, +0.002), (+0.078, +0.346), (+0.030, +0.215), (-0.197, -0.108)]
    ci_c4 = [(+0.020, +0.105), (+0.253, +0.505), (+0.123, +0.287), (-0.090, -0.016)]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.26
    xs = range(4)
    b1 = ax.bar([x - w for x in xs], prompt, w, color=C_PROMPT, label=t["prompt"])
    b2 = ax.bar(list(xs), c3, w, color=C_STATE_LIGHT, label=t["c_c3"])
    b3 = ax.bar([x + w for x in xs], c4, w, color=C_STATE, label=t["c_c4"])
    for bars, vals, ci in ((b1, prompt, ci_prompt), (b2, c3, ci_c3), (b3, c4, ci_c4)):
        for b, v, (lo, hi) in zip(bars, vals, ci):
            x = b.get_x() + b.get_width() / 2
            ax.plot([x, x], [lo, hi], color="#333333", lw=1.2, zorder=4)
            ax.plot([x - 0.03, x + 0.03], [lo, lo], color="#333333", lw=1.2, zorder=4)
            ax.plot([x - 0.03, x + 0.03], [hi, hi], color="#333333", lw=1.2, zorder=4)
            ax.text(x, (hi if v >= 0 else lo) + (0.012 if v >= 0 else -0.012), "{:+.2f}".format(v) if abs(v) >= 0.005 else "0.00",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="#555555", lw=1.2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(t["c_cats"])
    ax.set_ylabel(t["delta_acc"])
    ax.set_ylim(-0.30, 0.58)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.set_title(t["c_title"], fontsize=12, loc="left")
    ax.text(0.0, -0.13, t["c_note"], transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#555555")
    save(fig, lang, "03_facts")


# ---------------------------------------------------------------------------
# 04. Form transfer (docs/results.md, "Form transfer" and "Generation: 1.5B")
# ---------------------------------------------------------------------------
def fig_transfer(lang):
    t = S[lang]
    data = {  # (values, intervals) for prompt and state; bootstrap over the 40 units
        "t_sub04": (([-0.002, -0.221], [(-0.052, +0.050), (-0.277, -0.163)]),
                    ([+0.242, -0.008], [(+0.160, +0.325), (-0.083, +0.067)])),
        "t_sub15": (([-0.031, -0.010], [(-0.085, +0.019), (-0.056, +0.033)]),
                    ([+0.046, +0.102], [(-0.010, +0.100), (+0.052, +0.152)])),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    w = 0.34
    for ax, (key, ((pr, pci), (st, sci))) in zip(axes, data.items()):
        b1 = ax.bar([x - w / 2 for x in range(2)], pr, w, color=C_PROMPT, label=t["prompt"])
        b2 = ax.bar([x + w / 2 for x in range(2)], st, w, color=C_STATE, label=t["state"])
        for bars, vals, ci in ((b1, pr, pci), (b2, st, sci)):
            for b, v, (lo, hi) in zip(bars, vals, ci):
                x = b.get_x() + b.get_width() / 2
                ax.plot([x, x], [lo, hi], color="#333333", lw=1.2, zorder=4)
                ax.plot([x - 0.03, x + 0.03], [lo, lo], color="#333333", lw=1.2, zorder=4)
                ax.plot([x - 0.03, x + 0.03], [hi, hi], color="#333333", lw=1.2, zorder=4)
                ax.text(x + 0.05, (hi if v >= 0 else lo) + (0.012 if v >= 0 else -0.012), "{:+.2f}".format(v) if abs(v) >= 0.005 else "0.00",
                        ha="left", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.axhline(0, color="#555555", lw=1.2)
        ax.set_xticks(range(2))
        ax.set_xticklabels(t["t_cats"])
        ax.set_title(t[key], fontsize=11, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(t["delta_acc"])
    axes[0].set_ylim(-0.33, 0.36)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(t["t_title"], fontsize=12, x=0.01, ha="left")
    fig.text(0.01, -0.02, {"en": "whiskers: 95 % interval, bootstrap over the 40 units", "ru": "усы: 95 %-й интервал, бутстрэп по 40 единицам"}[lang], fontsize=8.5, color="#555555")
    fig.tight_layout()
    save(fig, lang, "04_form_transfer")


# ---------------------------------------------------------------------------
# 05. Six prompts (docs/results.md, "Scale × quantization" and "Five other prompts")
# ---------------------------------------------------------------------------
PROMPTS = {  # per model: [(prompt delta, state delta)] in the order of p_names
    "0.4B": [(-0.143, +0.057), (-0.093, +0.078), (-0.095, +0.051), (-0.072, +0.076), (-0.012, +0.002), (-0.119, -0.054)],
    "1.5B": [(-0.039, +0.038), (-0.032, +0.049), (-0.027, +0.018), (-0.043, +0.021), (+0.023, +0.048), (-0.028, +0.038)],
}


def fig_prompts(lang):
    t = S[lang]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    ys = list(range(6))[::-1]
    for ax, (model, rows) in zip(axes, PROMPTS.items()):
        for y, (dp, ds) in zip(ys, rows):
            ax.plot([dp, ds], [y, y], color="#c9ccd1", lw=2, zorder=1)
            ax.scatter([dp], [y], s=65, color=C_PROMPT, zorder=3)
            ax.scatter([ds], [y], s=65, color=C_STATE, zorder=3)
        ax.axvline(0, color="#555555", lw=1.2)
        ax.set_xlim(-0.17, 0.11)
        ax.set_title("g1d " + model, fontsize=11, loc="left")
        ax.set_xlabel(t["p_x"])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(t["p_names"])
    # annotate the one case where the state is below the no-prompt model
    fig.text(0.01, -0.02, "* " + t["p_note"], fontsize=8.5, color="#555555")
    axes[0].text(-0.054, -0.38, "*", ha="center", va="center", fontsize=12, color=C_STATE)
    axes[1].scatter([], [], s=65, color=C_PROMPT, label=t["prompt"])
    axes[1].scatter([], [], s=65, color=C_STATE, label=t["state"])
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(t["p_title"], fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, lang, "05_prompts")


# ---------------------------------------------------------------------------
# 06. Provocations (docs/results.md, "Identity provocations")
# ---------------------------------------------------------------------------
def fig_prov(lang):
    t = S[lang]
    cols = [C_NONE, C_PROMPT, C_STATE, "#6d9be6", "#a9c3ee", "#12408f"]
    d04 = [0.61, 0.34, 0.21, None, 0.24, 0.02]
    d15 = [0.85, 0.57, 0.62, 0.54, 0.47, 0.36]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, title, vals in ((axes[0], "g1d 0.4B bf16", d04), (axes[1], "g1d 1.5B bf16", d15)):
        idx = [i for i, v in enumerate(vals) if v is not None]
        pos = list(range(len(idx)))
        bars = ax.bar(pos, [vals[i] for i in idx], 0.65, color=[cols[i] for i in idx])
        label_bars(ax, bars, fmt="{:.2f}", size=9)
        ax.set_xticks(pos)
        ax.set_xticklabels([t["v_short"][i].replace(" ", "\n") for i in idx], fontsize=9)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel({"en": "share of answers", "ru": "доля ответов"}[lang])
    fig.text(0.01, -0.01, {"en": "c4 = trained state;  c4c = c4 without 8 answers that taught a foreign creator;  "
                                    "c4pd = + self-distilled refusals;  c4ph = + hand-written refusals",
                              "ru": "c4 — обученное состояние;  c4c — c4 без 8 ответов, учивших чужому создателю;  "
                                    "c4pd — + отказы самодистилляцией;  c4ph — + ручные отказы"}[lang],
                 fontsize=8, color="#555555")
    fig.suptitle(t["v_title"], fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.905, t["v_sub"], fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, lang, "06_provocations")


# ---------------------------------------------------------------------------
# 07. Dialogues (docs/results.md, "Role over time")
# ---------------------------------------------------------------------------
def fig_dialogues(lang):
    t = S[lang]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax = axes[0]
    turns = [1, 5, 10]
    ax.plot(turns, [1.00, 0.75, 0.44], marker="o", color=C_PROMPT, label=t["prompt"], lw=2)
    ax.plot(turns, [1.00, 0.85, 0.58], marker="o", color=C_STATE_LIGHT, label=t["d_state"], lw=2)
    ax.plot(turns, [1.00, 0.96, 0.60], marker="o", color=C_STATE, label=t["d_pc"], lw=2)
    ax.set_xticks(turns)
    ax.set_xlabel(t["d_turn"])
    ax.set_ylim(0.3, 1.05)
    ax.set_title(t["d_role"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax = axes[1]
    ax.plot([1, 10], [0.84, 0.88], marker="o", color=C_NONE, label=t["d_np"], lw=2)
    ax.plot([1, 10], [0.81, 0.53], marker="o", color=C_PROMPT, label=t["prompt"], lw=2)
    ax.plot([1, 10], [0.83, 0.74], marker="o", color=C_STATE, label=t["d_state"], lw=2)
    ax.set_xticks([1, 10])
    ax.set_xlabel(t["d_turn"])
    ax.set_ylim(0.3, 1.05)
    ax.set_title(t["d_facts"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle(t["d_title"], fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, lang, "07_dialogues")


# ---------------------------------------------------------------------------
# 08. Classification (docs/results.md, "Role over time" / classification)
# ---------------------------------------------------------------------------
def fig_cls(lang):
    t = S[lang]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    toks = ["~100", "~400", "~1000"]
    x = range(3)
    # a) 0.4B
    ax = axes[0]
    w = 0.2
    series = [(t["prompt"], [0.001, 0.0, 0.0], C_PROMPT), (t["state"], [0.843, 0.894, 0.883], C_STATE),
              (t["k_lora"], [0.890, 0.917, 0.910], C_LORA), (t["k_top"], [0.906, 0.903, 0.917], C_TOP4)]
    for i, (name, vals, col) in enumerate(series):
        ax.bar([xx + (i - 1.5) * w for xx in x], vals, w, color=col, label=name)
    ax.set_xticks(list(x)); ax.set_xticklabels(toks)
    ax.set_xlabel(t["k_tok"]); ax.set_ylabel(t["k_acc"]); ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(t["k_a"], fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2, columnspacing=1.0)
    # b) 1.5B
    ax = axes[1]
    w = 0.26
    series = [(t["prompt"], [0.318, 0.263, 0.207], C_PROMPT), (t["k_dist"], [0.766, 0.724, 0.645], C_STATE_LIGHT),
              (t["k_gold"], [0.869, 0.889, 0.874], C_STATE)]
    for i, (name, vals, col) in enumerate(series):
        ax.bar([xx + (i - 1) * w for xx in x], vals, w, color=col, label=name)
    ax.set_xticks(list(x)); ax.set_xticklabels(toks)
    ax.set_xlabel(t["k_tok"]); ax.set_ylim(0, 1.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(t["k_b"], fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    # c) data curve
    ax = axes[2]
    n = [25, 50, 125, 250]
    ax.plot(n, [0.801, 0.851, 0.899, 0.894], marker="o", color=C_STATE, lw=2, label=t["state"])
    ax.plot(n, [0.856, 0.892, 0.912, 0.917], marker="o", color=C_LORA, lw=2, label=t["k_lora"])
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels([str(v) for v in n])
    ax.minorticks_off()
    ax.set_xlabel(t["k_n"]); ax.set_ylim(0.7, 1.0)
    ax.set_title(t["k_c"], fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle(t["k_title"], fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, lang, "08_classification")


# ---------------------------------------------------------------------------
# 09. Cost (docs/results.md, "Cost")
# ---------------------------------------------------------------------------
def fig_cost(lang):
    t = S[lang]
    models = ["0.4B", "1.5B", "2.9B"]
    short_text, short_fold = [57, 157, 360], [48, 131, 301]
    long_text, long_fold = [618, 1956, 4259], [53, 146, 337]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, a, b, title in ((axes[0], short_text, short_fold, t["x_short"]), (axes[1], long_text, long_fold, t["x_long"])):
        w = 0.36
        b1 = ax.bar([i - w / 2 for i in range(3)], a, w, color=C_PROMPT, label=t["x_text"])
        b2 = ax.bar([i + w / 2 for i in range(3)], b, w, color=C_STATE, label=t["x_fold"])
        for bars in (b1, b2):
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() * 1.01, f"{r.get_height():.0f}",
                        ha="center", va="bottom", fontsize=8)
        for i, (u, v) in enumerate(zip(a, b)):
            ax.text(i, max(u, v) + 0.09 * max(a), f"×{u / v:.1f}", ha="center", va="bottom", fontsize=9, color=C_STATE, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(models)
        ax.set_ylim(0, max(a) * 1.3)
        ax.set_title(title, fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(t["x_ms"])
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(t["x_title"], fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, lang, "09_cost")


if __name__ == "__main__":
    for lang in ("en", "ru"):
        fig_overview(lang)
        fig_matrix(lang)
        fig_c3c4(lang)
        fig_transfer(lang)
        fig_prompts(lang)
        fig_prov(lang)
        fig_dialogues(lang)
        fig_cls(lang)
        fig_cost(lang)
    print("done")
