[English](REPRODUCE.md) · [Русский](REPRODUCE.ru.md)

# Reproduction

Everything was computed on a MacBook Air M4, 16 GB, macOS. Scripts are run **from the repository root** (except the
classification data preparation — see below); paths inside the scripts are relative. Summaries and comments inside
`experiments/` are in Russian.

## 1. What you need

- **gen-run** from [SwiftRWKV](https://github.com/RafaelUI/SwiftRWKV), commit `c332453`:

  ```sh
  git clone https://github.com/RafaelUI/SwiftRWKV && cd SwiftRWKV && git checkout c332453
  swift build -c release --product gen-run
  export GEN_RUN=$PWD/.build/release/gen-run      # mlx-swift_Cmlx.bundle must sit next to it — the build does that
  ```

  The stages of 27 Oct – 3 Nov were computed with several commits (`c689590` → `3106b2d` → `148861d` → `1057963` →
  `a112626` → `c332453`); every next binary was checked against the answers of the previous one — on the old arms the answers
  are **bit for bit** the same (the gates are in `experiments/*/gate/`). So `c332453` reproduces all stages from 27 Oct. The
  stages of 19–23 Oct (`experiments/10{19,20,22}_*`) were computed with earlier commits and are kept as they are, without such
  a check.
- **Python 3 + numpy** — only for summaries and data preparation.
- **Models** — in `models/` (see [models/README.md](models/README.md); checksums in `models/SHA256SUMS`).
- For stage 1027 (g1d quantization) — [rwkv-quant](https://github.com/RafaelUI/rwkv-quant) `d228746` and rwkv-metal (variable
  `RWKV_QUANT`, default `../rwkv-quant`).

## 2. Ready answers instead of recomputing

Every stage folder holds `outputs.tar.gz` — all model answers (`*.json`), teacher caches and training logs:

```sh
for d in experiments/*/; do [ -f $d/outputs.tar.gz ] && tar xzf $d/outputs.tar.gz -C $d; done
```

After unpacking, all summaries are recomputed without the model (checked: they match the `*_summary.md` in the repository):

```sh
python3 experiments/1027_matrix/summ.py experiments/1027_matrix/m/g1d_0p4b_bf16
python3 experiments/1027_matrix/axis.py axis.md experiments/1027_matrix/m/g1d_{0p4b,1p5b}_{bf16,red,comp}
python3 experiments/1029_prompts/ptable.py
python3 experiments/1030_provocations_control/provsum.py
python3 experiments/1030_provocations_control/ctlsum.py
python3 experiments/1031_prov_training/ptsum.py g1d_1p5b_bf16 c4c c4pd c4ph
python3 experiments/1101_dialogues/mtsum.py
python3 experiments/1101_classification/clssum.py                 # 0.4B
python3 experiments/1101_classification/clssum2.py g1d_1p5b_bf16  # 1.5B
python3 experiments/1101_classification/curvesum.py
python3 experiments/1103_cost/ttftsum.py
```

The run scripts skip files that already exist — to recompute an arm, remove its file; to recompute from scratch, do not
unpack `outputs.tar.gz`. The teacher caches (`teacher*.json`) from the archive give exactly the same training data.

## 3. Order of the stages

| stage | what | how to run |
|---|---|---|
| [1027_matrix](experiments/1027_matrix) | scale × quantization, protocol c4 and variant c | `bash experiments/1027_matrix/matrix.sh` |
| [1029_prompts](experiments/1029_prompts) | five other prompts | `python3 experiments/1029_prompts/prompts.py build experiments/1029_prompts/p`, `bash experiments/1029_prompts/run_prompts.sh` |
| [1030_provocations_control](experiments/1030_provocations_control) | provocations; strong control (top4 / LoRA) | `run_prov.sh`; `run_ctl.sh` |
| [1031_prov_training](experiments/1031_prov_training) | provocations in training | `run_teacher.sh` → `curate.py ids <model>` → `build_caches.py` → `curate.py drop` (the numbers are in `curate_log.md`) → `STAGE=... run_pt.sh` |
| [1101_dialogues](experiments/1101_dialogues) | role over time, control on facts | `run_mt.sh`, `run_mt_facts.sh` |
| [1101_classification](experiments/1101_classification) | news classification, data curve | preparation (below), then `run_cls.sh`, `run_cls15.sh`, `run_curve.sh` |
| [1103_cost](experiments/1103_cost) | time to first token | `run_ttft.sh` (on an idle machine) |
| [1102_audit](experiments/1102_audit) | manual check of the regular expressions | `judged.json` — manual judgements |

The teacher (the same model with the system string) is computed on the first state training and cached.

## 4. Classification data

The Lenta.ru texts are not in the repository (the dataset has no stated licence). They are rebuilt from
[data-silence/lenta.ru_2](https://huggingface.co/datasets/data-silence/lenta.ru_2), bit for bit (checked):

```sh
cd experiments/1101_classification
curl -LO https://huggingface.co/datasets/data-silence/lenta.ru_2/resolve/main/final_lenta_new.csv.gz
python3 build.py && python3 build_val.py && python3 make_gold.py && python3 make_curve.py
python3 select_urls.py      # the URL list of the selected articles — also in lenta_selection.tsv
cd ../..
```

Test and validation labels without texts — `test_labels.tsv`, `val_labels.tsv` (the summaries read them); in the model
answers inside `outputs.tar.gz` the article text is replaced by a number (`test_mid:17`, `train:3`, …).

## 5. Time and memory

Guidelines for an M4 16 GB: training a 0.4B state — ~5 min (Neuron) / ~17 min (articles up to 1000 tokens), 1.5B — 16 / 51
min, 2.9B COMPRESSION — 31 min; peaks 1.6–5.4 GB. The full 1027 matrix takes about a day; 1.5B classification — ~10 h. The
summary table — [experiments/1103_cost/cost_table.md](experiments/1103_cost/cost_table.md).

## 6. Figures

The pictures in the README and in `docs/` are drawn by `python3 figures/make_figures.py` (needs matplotlib) from the numbers
in [docs/results.md](docs/results.md); `figures/en/` and `figures/ru/` hold the same plots with English and Russian labels.
