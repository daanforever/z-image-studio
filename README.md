# Z-Image-Turbo Studio

Локальный веб-интерфейс на **Gradio** для инференса [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) через `diffusers`. Это не обёртка над `zimg.exe`: модель грузится напрямую, с CUDA, если PyTorch собран с GPU.

Turbo-рецепт по документации модели: **9 шагов** (8 проходов DiT) и **`guidance_scale=0`**.

## Что умеет

- Промпт, пресеты разрешения, seed, шаги, time shift
- Путь к модели: Hugging Face id **или** локальный snapshot
- Автовыбор `cuda` / `cpu`, статус VRAM
- Precision: **fp8** по умолчанию; также `bfloat16` / `float16` / `float32` / `int8` (torchao на DiT)
- CPU offload и VAE tiling, если мало памяти
- Сохранение PNG в `outputs/`
- Демо-режим без весов и без GPU (чтобы открыть UI)

Интерфейс не использует квантованные `Disty0/Z-Image-Turbo-SDNQ-*`. Нужен официальный **Z-Image-Turbo**.

## Установка (Windows, RTX 50xx)

В каталоге проекта:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

`pip install torch` с PyPI ставит **CPU-сборку**. На RTX 5080 её недостаточно — нужна колесо `cu128`/`cu130` с `download.pytorch.org`. Если CPU-версия уже стоит, обычный `pip install` ничего не меняет: нужен `--force-reinstall` (или сначала `pip uninstall torch torchvision`).

Если `from diffusers import ZImagePipeline` падает, поставьте diffusers из git:

```bat
pip install git+https://github.com/huggingface/diffusers
```

Проверка GPU:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Ожидается `True` и имя видеокарты.

## Запуск

Самый простой путь на вашей машине — `launch.bat`. Он выставляет:

| Переменная | Значение по умолчанию |
|---|---|
| `HF_HUB_CACHE` | `E:\Backup\huggingface\hub` |
| `HF_HOME` | `E:\Backup\huggingface` |
| `HF_HUB_OFFLINE` | `0` (можно докачивать с Hub) |
| `ZIMAGE_MODEL` | `Tongyi-MAI/Z-Image-Turbo` |
| порт | `43127` |

```bat
launch.bat
```

Либо вручную:

```bat
python app.py --host 127.0.0.1 --port 43127
```

Откройте http://127.0.0.1:43127

Чтобы грузить конкретный snapshot, не трогая Hub:

```bat
set ZIMAGE_MODEL=E:\Backup\huggingface\hub\models--Tongyi-MAI--Z-Image-Turbo\snapshots\0e36c2b379e66fa531d01cc531c44919e5f1c6fd
python app.py
```

Можно скопировать `.env.example` в `.env` — приложение подхватит переменные при старте.

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py --host 127.0.0.1 --port 43127
```

Без CUDA генерация на CPU будет очень медленной. Для проверки интерфейса:

```bash
ZIMAGE_DEMO=1 python app.py --port 43127
```

## Параметры Turbo vs полная Z-Image

| | Z-Image-Turbo | Z-Image |
|---|---|---|
| Модель | `Tongyi-MAI/Z-Image-Turbo` | `Tongyi-MAI/Z-Image` |
| Шаги | 9 | 28–50 |
| Guidance | 0.0 | 3.0–5.0 |
| Negative prompt | не нужен | полезен |

Это приложение заточено под Turbo. Полную Z-Image можно указать в поле модели, но guidance тогда поднимите вручную в блоке «Дополнительно».

## VRAM

По умолчанию Turbo грузится в **fp8** (Ada 8.9+ / Blackwell, в том числе RTX 5080; не сочетается с CPU offload). **bfloat16** — полный режим, если fp8 не нужен. **int8** — если нужен CPU offload. Плюс **VAE tiling**. Это не Disty0/SDNQ-чекпоинты: квантуется тот же `Tongyi-MAI/Z-Image-Turbo`.

## Тесты

```bat
pip install -r requirements-dev.txt
pytest
```

Покрыты конфиг, runtime/demo/pipeline и обработчики UI. Веса модели и живой GPU не требуются.

## Структура

```
app.py                 точка входа (python app.py)
zimage/config.py       пресеты и .env
zimage/engine/         статус устройства, demo-кадр, пайплайн
zimage/ui/             тема, статус, обработчики, вёрстка Gradio
tests/                 pytest
launch.bat             запуск под Windows с вашими путями
outputs/               сохранённые PNG
```
