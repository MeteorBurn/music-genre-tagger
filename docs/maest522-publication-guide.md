# MAEST 522: публикация на Hugging Face

> Метка: upload намеренно отделён от build. По умолчанию команда только
> проверяет локальный release и не делает сетевых изменений.

## 1. Что публикуется

Model repository содержит только allowlisted artifacts:

- standard `ASTForAudioClassification` — `config.json` + `model.safetensors`;
- `native-maest-522.ckpt` с обоими MAEST heads;
- `labels-522.txt`;
- exact remote-code frontend `feature_extraction_maest.py` и
  `preprocessor_config.json`;
- model card, evaluation/parity/provenance reports и `SHA256SUMS`.

Аудио, локальные paths, SQLite, teacher cache, M3U и приватные annotation notes
в staging запрещены. Dataset repository, если он создаётся, публикует только
metadata/annotations и не заявляет права на распространение аудио.

## 2. License gate

До документированной проверки grant для точного исходного checkpoint model card
использует `license: other`. Лицензия source code, происхождение weights, права
на annotations и права на audio рассматриваются отдельно. Публичную visibility
до завершения этой проверки включать нельзя.

## 3. Локальная проверка

```powershell
& '.\.venv\Scripts\python.exe' -m tools.maest522.hub_publish `
  --release-dir '.\release\discogs-maest-30s-pw-129e-522l'
```

Проверяются allowlist, каждый SHA256SUMS, `license: other`, test/evaluation gates
и parity минимум на 30 окнах из 10 треков. `HfApi` при этом не создаётся.

## 4. Private upload

Авторизация использует стандартный Hugging Face token mechanism. Token нельзя
передавать аргументом команды или записывать в config/log:

```powershell
hf auth login
hf auth whoami
```

Первый upload всегда рекомендуется private:

```powershell
& '.\.venv\Scripts\python.exe' -m tools.maest522.hub_publish `
  --release-dir '.\release\discogs-maest-30s-pw-129e-522l' `
  --repo-id 'ORG/discogs-maest-30s-pw-129e-522l' `
  --version '0.1.0' `
  --push
```

После upload нужно вручную проверить rendered model card, список файлов,
provenance/hashes и private smoke test. Команда ничего не удаляет из remote repo.

`--public` разрешён только после отдельного license review и явного решения:

```powershell
& '.\.venv\Scripts\python.exe' -m tools.maest522.hub_publish `
  --release-dir '.\release\discogs-maest-30s-pw-129e-522l' `
  --repo-id 'ORG/discogs-maest-30s-pw-129e-522l' `
  --version '0.1.0' `
  --push `
  --public
```

## 5. Standard Transformers inference

Raw 16 kHz mono audio требует custom exact frontend. Generic AST feature
extractor не является эквивалентной заменой:

```python
import torch
from transformers import ASTForAudioClassification, AutoFeatureExtractor

repo_id = "ORG/discogs-maest-30s-pw-129e-522l"
extractor = AutoFeatureExtractor.from_pretrained(
    repo_id,
    trust_remote_code=True,
)
model = ASTForAudioClassification.from_pretrained(repo_id).eval()

inputs = extractor(waveform_16k_mono, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    probabilities = torch.sigmoid(model(**inputs).logits)
```

Если mel уже получен штатным native MAEST frontend в форме
`[batch, 96, time]`, standard AST принимает только transpose:

```python
hf_input = native_mel.transpose(1, 2).contiguous()
with torch.no_grad():
    probabilities = torch.sigmoid(model(input_values=hf_input).logits)
```

## 6. Native inference

```python
from pathlib import Path
from tools.maest522.native_model import get_maest_522, predict_522

model = get_maest_522(Path("native-maest-522.ckpt"), device="cuda")
probabilities = predict_522(model, native_mel.to("cuda"))
```

Оба пути возвращают один и тот же ordered 522-vector; sigmoid применяется ровно
один раз. Для трека используются 20%/50%/80% окна и mean aggregation probabilities.
