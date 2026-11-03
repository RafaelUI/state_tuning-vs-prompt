[English](REPRODUCE.md) · *Русский*

# Воспроизведение

Всё считалось на MacBook Air M4, 16 ГБ, macOS. Скрипты запускаются **из корня репозитория** (кроме подготовки данных
классификации — см. ниже); пути внутри скриптов относительные.

## 1. Что нужно

- **gen-run** из [SwiftRWKV](https://github.com/RafaelUI/SwiftRWKV), коммит `c332453`:

  ```sh
  git clone https://github.com/RafaelUI/SwiftRWKV && cd SwiftRWKV && git checkout c332453
  swift build -c release --product gen-run
  export GEN_RUN=$PWD/.build/release/gen-run      # рядом должен лежать mlx-swift_Cmlx.bundle — так и собирается
  ```

  Этапы 27.10–03.11 считались несколькими коммитами (`c689590` → `3106b2d` → `148861d` → `1057963` → `a112626` →
  `c332453`); каждый следующий бинарник проверялся на ответах предыдущего — на старых плечах ответы **побитно** те же
  (гейты лежат в `experiments/*/gate/`). Поэтому `c332453` воспроизводит все этапы с 27.10. Этапы 19–23.10
  (`experiments/10{19,20,22}_*`) считались более ранними коммитами и сохранены как есть, без такой проверки.
- **Python 3 + numpy** — только для сводок и подготовки данных.
- **Модели** — в `models/` (см. [models/README.md](models/README.ru.md), контрольные суммы — `models/SHA256SUMS`).
- Для этапа 1027 (квантизация g1d) — [rwkv-quant](https://github.com/RafaelUI/rwkv-quant) `d228746` и rwkv-metal
  (переменная `RWKV_QUANT`, по умолчанию `../rwkv-quant`).

## 2. Готовые ответы вместо пересчёта

В каждой папке этапа лежит `outputs.tar.gz` — все ответы модели (`*.json`), кэши учителя и журналы обучения:

```sh
for d in experiments/*/; do [ -f $d/outputs.tar.gz ] && tar xzf $d/outputs.tar.gz -C $d; done
```

После распаковки все сводки пересчитываются без модели (проверено: совпадают с `*_summary.md` в репозитории):

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

Скрипты прогонов пропускают уже готовые файлы — чтобы пересчитать плечо, уберите его файл; для пересчёта с нуля не
распаковывайте `outputs.tar.gz`. Кэши учителя (`teacher*.json`) из архива дают ровно те же обучающие данные.

## 3. Порядок этапов

| этап | что | запуск |
|---|---|---|
| [1027_matrix](experiments/1027_matrix) | масштаб × квантизация, протокол c4 и вариант c | `bash experiments/1027_matrix/matrix.sh` |
| [1029_prompts](experiments/1029_prompts) | пять других промптов | `python3 experiments/1029_prompts/prompts.py build experiments/1029_prompts/p`, `bash experiments/1029_prompts/run_prompts.sh` |
| [1030_provocations_control](experiments/1030_provocations_control) | провокации; сильный контроль (top4 / LoRA) | `run_prov.sh`; `run_ctl.sh` |
| [1031_prov_training](experiments/1031_prov_training) | провокации в обучении | `run_teacher.sh` → `curate.py ids <модель>` → `build_caches.py` → `curate.py drop` (номера — в `curate_log.md`) → `STAGE=... run_pt.sh` |
| [1101_dialogues](experiments/1101_dialogues) | роль во времени, контроль на фактах | `run_mt.sh`, `run_mt_facts.sh` |
| [1101_classification](experiments/1101_classification) | классификация новостей, кривая по данным | подготовка (ниже), затем `run_cls.sh`, `run_cls15.sh`, `run_curve.sh` |
| [1103_cost](experiments/1103_cost) | время до первого токена | `run_ttft.sh` (на свободной машине) |
| [1102_audit](experiments/1102_audit) | ручная сверка регэкспов | `judged.json` — оценки вручную |

Учитель (та же модель с системной строкой) считается при первом обучении состояния и кэшируется.

## 4. Данные классификации

Тексты Lenta.ru в репозиторий не положены (у набора не указана лицензия). Они собираются заново из
[data-silence/lenta.ru_2](https://huggingface.co/datasets/data-silence/lenta.ru_2), побитно (проверено):

```sh
cd experiments/1101_classification
curl -LO https://huggingface.co/datasets/data-silence/lenta.ru_2/resolve/main/final_lenta_new.csv.gz
python3 build.py && python3 build_val.py && python3 make_gold.py && python3 make_curve.py
python3 select_urls.py      # список URL отобранных статей — он же в lenta_selection.tsv
cd ../..
```

Метки проверки и валидации без текстов — `test_labels.tsv`, `val_labels.tsv` (их читают сводки); в ответах моделей
в `outputs.tar.gz` текст статьи заменён номером (`test_mid:17`, `train:3`, …).

## 5. Время и память

Ориентиры для M4 16 ГБ: обучение состояния 0.4B — ~5 мин (Neuron) / ~17 мин (статьи до 1000 токенов), 1.5B — 16 / 51 мин,
2.9B COMPRESSION — 31 мин; пики 1.6–5.4 ГБ. Полная матрица 1027 — около суток; классификация 1.5B — ~10 ч.
Сводная таблица — [experiments/1103_cost/cost_table.md](experiments/1103_cost/cost_table.md).
