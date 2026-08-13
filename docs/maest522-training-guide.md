# MAEST 522: подготовка и hybrid continual fine-tuning

> Метка: это operational guide для community-extension. Исходный runtime
> `src/` и default-модель `maest_519l_pytorch` не изменяются.

## 1. Что именно обучается

Расширение остаётся одной нативной MAEST-моделью. Порядок outputs фиксирован:

```text
0..518   исходные Discogs labels из maest-infer==0.2.0
519      Electronic---Minimal-Deep-Tech
520      Electronic---Microhouse
521      Electronic---RoMinimal
```

Во время обучения legacy 519 rows и три extension rows представлены разными
модулями, чтобы AdamW не менял старые rows через weight decay. При экспорте они
снова объединяются в стандартные `head.1` и `head_dist` размером 522.

## 2. Установка training-зависимостей

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r '.\requirements-training.txt'
```

Annotation UI использует отдельный `requirements-annotation.txt`.

## 3. Сначала зафиксировать датасет

До обучения нужно последовательно собрать подтверждённые `positive/negative`
примеры для всех трёх новых классов, выполнить fingerprint audit и один общий
group-disjoint split 70/15/15. По умолчанию цель каждого класса —
`1000 positive + 1000 hard negatives`, но её можно независимо изменить до
freeze. Экспорт UI создаёт `training.jsonl` и `dataset_summary.json`.

Manifest строится из последних событий confirmed-label journal, а не из папок
кандидатов или очередей. Для каждого трека записываются все три состояния и
`label_mask`: `positive/negative → 1`, `uncertain/unreviewed → 0`. Поэтому один
и тот же трек может обучать один extension output и быть masked для двух других.

Portable `audio_ref` разрешается относительно папки manifest. Поэтому рядом с
manifest нужно создать каталог `audio/` и поместить туда файлы с именами,
указанными в `audio_ref`. Можно вместо этого экспортировать локальный manifest
с абсолютным `path`, но его нельзя публиковать.

Отдельно создаются два непересекающихся replay split:

- `replay_train` — участвует только в teacher distillation;
- `regression_holdout` — никогда не обновляет веса и используется для gates.

Любое изменение manifest, teacher checkpoint или preprocessing version
инвалидирует cache/resume по SHA-256.

## 4. Checkpoint surgery 519 → 522

```powershell
& '.\.venv\Scripts\python.exe' -m tools.maest522.checkpoint `
  '.\src\models\discogs-maest-30s-pw-129e-519l-swa.ckpt' `
  '.\training-artifacts\expanded-init' `
  --seed 522 `
  --prior-probability 0.01
```

Получаются:

- `expanded-init.ckpt`;
- `labels-522.txt`;
- `expansion-report.json` с source/output SHA-256.

Конвертер принимает только точные формы `(519,768)` / `(519,)` и гарантирует
bit-identical первые 519 rows всех четырёх classifier tensors.

Проверка реального forward/backward без записи model artifacts:

```powershell
& '.\.venv\Scripts\python.exe' -m tools.maest522.train `
  --dry-run-checkpoint '.\src\models\discogs-maest-30s-pw-129e-519l-swa.ckpt'
```

## 5. Audio windows и teacher cache

`training_data.py` повторяет production contract:

- mono 16 kHz;
- окно 30 секунд;
- центры 20%, 50%, 80%;
- clamp у границ и удаление повторных offsets;
- штатный `maest_infer.helpers.melspectrogram.MelSpectrogram`.

Ключ cache:

```text
track_id + offset_ms + teacher_sha256 + preprocessing_version
```

Каждая запись safetensors содержит 519 teacher logits и pooled embedding 768.
SQLite index и SHA-256 файла проверяются при каждом чтении.

## 6. Три стадии обучения

### Stage 1 — `extension_heads`

Обучаются только `extension_head` и `extension_dist_head`. Backbone, final norm
и legacy heads заморожены.

### Stage 2 — `blocks_10_11`

Дополнительно размораживаются blocks 10–11 и `backbone.norm`. Рекомендуемые
стартовые learning rates: `3e-4` для heads и `1e-5` для backbone.

### Stage 3 — `blocks_8_11`

Размораживаются также blocks 8–9. Эту стадию запускают только после plateau
Stage 2, когда проблема не объясняется ошибками разметки/нехваткой hard
negatives, а regression gates всё ещё проходят.

На границе stages optimizer создаётся заново. Resume внутри stage восстанавливает
model, optimizer, epoch/global step и состояния Python/NumPy/PyTorch RNG.

Objective:

```text
L = 1.0 * masked_BCE_new
  + 2.0 * SmoothL1(student_legacy, teacher_legacy)
  + 1e-4 * L2-SP(unfrozen_backbone, original_backbone)
  + 0.25 * masked_BCE_dist_head
```

Uncertain cells имеют mask 0. Hard negatives — явный target 0. Positive class
weights вычисляются только по `train`, ограничиваются диапазоном `[1,10]` и
сохраняются в run artifacts.

## 7. Артефакты run и возобновление

До первого optimizer step должны существовать:

- `resolved-config.json`;
- `environment.json`;
- `run-manifest.json` с input digests;
- затем `last.ckpt`, `best.ckpt`, `metrics.jsonl`.

Инициализация run из JSON-конфига:

```powershell
& '.\.venv\Scripts\python.exe' -m tools.maest522.train `
  --config '.\training-config.json' `
  --initialize-only
```

Trainer API `run_training_stage()` принимает epoch-batch factory, loss callback
и validation callback. Это позволяет не смешивать immutable dataset assembly с
экспериментальными sampling policies. Annotation/replay рекомендуется смешивать
1:1 по batch; SpecAugment применяется только к train.

## 8. Validation и locked test

Best checkpoint выбирается лексикографически:

1. regression gates должны пройти;
2. максимальный macro AP трёх новых labels;
3. затем macro F1 на validation-derived thresholds;
4. затем меньший legacy probability drift.

Release gates исходных 519 классов:

```text
mean absolute probability drift <= 0.01
mean top-10 overlap >= 0.90
teacher top-3 contained in student top-5 >= 0.98
mean pooled-embedding cosine >= 0.98
frequently-active class mean drift <= 0.03
frequently-active class p95 drift <= 0.10
```

Threshold каждого нового label выбирается на validation при максимальном F1 с
минимальной precision (по умолчанию 0.80), затем замораживается. Test запускается
ровно один раз:

```powershell
& '.\.venv\Scripts\python.exe' -m tools.maest522.evaluate `
  --run-dir '.\training-artifacts\run-001' `
  --split test `
  --report-json '.\training-artifacts\run-001\prepared-test-report.json' `
  --allow-test
```

Без `--allow-test` команда откажет. Повторный test-run для того же run directory
тоже блокируется.

## 9. Кандидаты для следующей версии датасета

Active-learning ranking принимает только training pool и до scoring исключает
reviewed tracks, val/test, siblings уже размеченных groups и исчерпанные quota.

```powershell
& '.\.venv\Scripts\python.exe' -m tools.maest522.active_learning `
  --candidates-json '.\training-artifacts\round-2-candidates.json' `
  --output '.\training-artifacts\round-2-acquisition.json' `
  --limit 300 `
  --seed 522
```

Основной output содержит только `track_id → acquisition_score`; predictions
остаются в отдельной diagnostic части и не показываются в annotation UI.
Результат импортируется в новый или ещё не замороженный annotation project.
Уже зафиксированные split/test и manifest текущей версии не дописываются.

## 10. Практический порядок для вашей библиотеки

1. В приложении кандидатов подготовить доверенные `+` и `-` M3U для
   `Minimal-Deep-Tech` и импортировать их через preflight/commit до текущей
   цели.
2. Вручную переключиться на `Microhouse`, затем на `RoMinimal`; UI сам жанр не
   меняет.
3. При необходимости дополнить каждый класс ручной одно-жанровой очередью.
4. Только после выполнения всех шести целей запустить fingerprints,
   зафиксировать один split и экспортировать masked manifest.
5. Проверить положительное и отрицательное покрытие каждого класса во всех
   трёх split и сохранить split-audit digest.
6. Собрать разнообразный `replay_train` из старых 519-классовых данных и
   отдельный `regression_holdout`.
7. Провести Stage 0 parity, затем Stage 1 и Stage 2.
8. Если нужны новые пограничные примеры, создать следующую версию annotation
   project/dataset; не дописывать уже замороженный test split.
9. Stage 3 применять только по условиям выше.
10. Зафиксировать thresholds, один раз открыть test и только затем готовить
    Hugging Face release.
