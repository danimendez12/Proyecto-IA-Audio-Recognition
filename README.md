# Proyecto-IA-Audio-Recognition

Audio classification project for 10 Speech Commands classes (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`).

## Speech Commands 10

Clasificador de los comandos `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop` y `go`. El modelo es una MobileNet-V1-style escrita con `torch.nn`, entrenada sobre mel-espectrogramas de 1 segundo y exportable a ONNX Runtime Mobile.

## Instalación y datos

```bash
python -m pip install -r requirements.txt
python data/download_and_prepare.py --root ./data --output-dir ./data/speech_commands_10
```

Construir el caché normalizado. La metadata registra el split oficial y la ruta del `.wav` para la augmentación de waveform:

```bash
python train.py --build-cache --augment-mode none --wandb-mode offline
```

El split reproducible es 70/15/15 sobre los clips originales y usa `--seed` (42 por defecto). La augmentación solo se aplica a los índices de train; validación y test permanecen limpios.

## Runs individuales

Modelo B base, sin augmentación:

```bash
python train.py --augment-mode none --checkpoint-path ./checkpoints/base/best_model.pt --wandb-mode online
```

Modelo B aumentado, con SpecAugment, time shifting y noise injection:

```bash
python train.py --augment-mode full --noise-dir ./data/speech_commands_10/_background_noise_ --checkpoint-path ./checkpoints/augmented/best_model.pt --wandb-mode online
```

`--augment-mode specaugment` aplica únicamente SpecAugment al caché. `--use-pitch-shift` y `--use-time-stretch` activan efectos opcionales de torchaudio y están desactivados por defecto por su mayor coste de CPU.

## Seis experimentos

Las mismas tres configuraciones se ejecutan para `base` y `augmented`. Cada run escribe `checkpoints/<run_name>/best_model.pt`; el resumen se guarda en `results/summary.csv` y como `wandb.Table`.

```bash
python run_experiments.py --wandb-mode online
```

Para ejecución local sin sincronización:

```bash
python run_experiments.py --wandb-mode offline
```

## Exportación ONNX

```bash
python export_onnx.py --checkpoint-path ./checkpoints/base/best_model.pt --cache-dir ./data/spectrogram_cache --onnx-path ./checkpoints/base/model.onnx
```

El script compara las salidas de PyTorch y ONNX Runtime con `np.testing.assert_allclose`.

## Resumen de arquitectura

```bash
python scripts/draw_architecture.py
```

La arquitectura usa un stem convolucional, tres bloques depthwise-separable, average pooling global, dropout y una capa lineal. `train.py` registra parámetros y MACs estimados en W&B.

## Documentación detallada

- [explicacion.md](explicacion.md): función de cada archivo, clase y función del repositorio.
- [WORK.md](WORK.md): guía práctica completa para ejecutar el pipeline y analizar los experimentos.

Install the pinned dependencies before running the pipeline:
```bash
python -m pip install -r requirements.txt
```

1. Download and filter dataset:
```bash
python data/download_and_prepare.py --root ./data --output-dir ./data/speech_commands_10
```

2. Train model (build spectrogram cache first):
```bash
python train.py --build-cache --augment-train --wandb-mode offline
```

The cache uses Speech Commands' official `validation_list.txt` and `testing_list.txt` files and stores globally normalized spectrograms. Rebuild it with `--build-cache` after changing preprocessing settings.

3. Export ONNX and verify outputs:
```bash
python export_onnx.py --checkpoint-path ./checkpoints/best_model.pt --cache-dir ./data/spectrogram_cache
```
