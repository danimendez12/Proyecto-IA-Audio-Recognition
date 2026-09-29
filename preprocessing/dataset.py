from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE
DEFAULT_N_MELS = 64

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}


@dataclass(frozen=True)
class CacheItem:
    spectrogram_path: str
    label: int


def _fix_length(waveform: Tensor) -> Tensor:
    if waveform.shape[1] > CLIP_SAMPLES:
        return waveform[:, :CLIP_SAMPLES]
    if waveform.shape[1] < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - waveform.shape[1]
        return F.pad(waveform, (0, pad))
    return waveform


def _create_mel_transform(sample_rate: int, n_mels: int) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=400,
        win_length=400,
        hop_length=160,
    )


def _wav_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for label in LABELS:
        files.extend(sorted((data_dir / label).glob("*.wav")))
    return files


def cache_mel_spectrograms(
    data_dir: str | Path,
    cache_dir: str | Path,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = DEFAULT_N_MELS,
) -> Path:
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    transform = _create_mel_transform(sample_rate=sample_rate, n_mels=n_mels)
    cache_items: list[CacheItem] = []

    for wav_path in _wav_files(data_dir):
        label_name = wav_path.parent.name
        if label_name not in LABEL_TO_INDEX:
            continue

        waveform, wav_sample_rate = torchaudio.load(wav_path)
        if wav_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, wav_sample_rate, sample_rate)

        waveform = _fix_length(waveform)
        mel_spec = transform(waveform)
        mel_spec = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)

        rel_output = Path(label_name) / f"{wav_path.stem}.pt"
        output_path = cache_dir / rel_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel_spec, output_path)

        cache_items.append(CacheItem(spectrogram_path=str(rel_output), label=LABEL_TO_INDEX[label_name]))

    metadata = {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "labels": LABELS,
        "items": [item.__dict__ for item in cache_items],
    }

    metadata_path = cache_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


class MelSpectrogramDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Cache metadata not found at {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.labels: list[str] = metadata["labels"]
        self.items: list[dict[str, int | str]] = metadata["items"]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        item = self.items[index]
        spectrogram_path = self.cache_dir / str(item["spectrogram_path"])
        spectrogram = torch.load(spectrogram_path, map_location="cpu").float()
        label = int(item["label"])
        return spectrogram, label
