# Proyecto-IA-Audio-Recognition

Audio classification project for 10 Speech Commands classes (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`).

## Quick start

1. Download and filter dataset:
```bash
python data/download_and_prepare.py --root ./data --output-dir ./data/speech_commands_10
```

2. Train model (build spectrogram cache first):
```bash
python train.py --build-cache --augment-train --wandb-mode offline
```

3. Export ONNX and verify outputs:
```bash
python export_onnx.py --checkpoint-path ./checkpoints/best_model.pt --cache-dir ./data/spectrogram_cache
```
