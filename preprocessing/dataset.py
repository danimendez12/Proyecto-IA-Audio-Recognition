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
    split: str
    wav_path: str = ""


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


def _official_splits(data_dir: Path) -> dict[str, str]:
    splits: dict[str, str] = {}
    for split, filename in (("val", "validation_list.txt"), ("test", "testing_list.txt")):
        split_path = data_dir / filename
        if not split_path.exists():
            raise FileNotFoundError(f"Official split file not found: {split_path}")
        for line in split_path.read_text(encoding="utf-8").splitlines():
            relative_path = line.strip()
            if relative_path:
                splits[relative_path] = split
    return splits


def _mel_spectrogram(wav_path: Path, transform: torchaudio.transforms.MelSpectrogram, sample_rate: int) -> Tensor:
    waveform, wav_sample_rate = torchaudio.load(wav_path)
    if wav_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, wav_sample_rate, sample_rate)
    waveform = _fix_length(waveform)
    mel_spec = transform(waveform)
    return torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0)


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
    split_map = _official_splits(data_dir)
    wav_paths = _wav_files(data_dir)

    total = 0
    value_sum = torch.tensor(0.0)
    squared_sum = torch.tensor(0.0)
    for wav_path in wav_paths:
        mel_spec = _mel_spectrogram(wav_path, transform, sample_rate)
        total += mel_spec.numel()
        value_sum += mel_spec.sum()
        squared_sum += mel_spec.square().sum()
    if total == 0:
        raise ValueError(f"No target .wav files found in {data_dir}")
    mean = value_sum / total
    std = (squared_sum / total - mean.square()).clamp_min(1e-12).sqrt()

    cache_items: list[CacheItem] = []

    for wav_path in wav_paths:
        label_name = wav_path.parent.name
        if label_name not in LABEL_TO_INDEX:
            continue

        mel_spec = (_mel_spectrogram(wav_path, transform, sample_rate) - mean) / std

        rel_output = Path(label_name) / f"{wav_path.stem}.pt"
        output_path = cache_dir / rel_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel_spec, output_path)

        relative_wav = wav_path.relative_to(data_dir).as_posix()
        cache_items.append(
            CacheItem(
                spectrogram_path=str(rel_output),
                label=LABEL_TO_INDEX[label_name],
                split=split_map.get(relative_wav, "train"),
                wav_path=relative_wav,
            )
        )

    metadata = {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "mean": float(mean),
        "std": float(std),
        "data_dir": str(data_dir.resolve()),
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
        self.metadata: dict[str, object] = metadata
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

    def split_indices(self) -> tuple[list[int], list[int], list[int]]:
        train: list[int] = []
        val: list[int] = []
        test: list[int] = []
        for index, item in enumerate(self.items):
            split = str(item.get("split", ""))
            if split == "train":
                train.append(index)
            elif split == "val":
                val.append(index)
            elif split == "test":
                test.append(index)
            else:
                raise ValueError(f"Unknown or missing split for cache item {index}: {split!r}")
        return train, val, test
