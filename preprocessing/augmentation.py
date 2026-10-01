from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torchaudio
from torch import Tensor, nn
from torch.utils.data import Dataset

from .dataset import CLIP_SAMPLES, SAMPLE_RATE, MelSpectrogramDataset, _create_mel_transform, _fix_length, _load_waveform


class SpecAugment(nn.Module):
    """Apply SpecAugment masks in normalized dB space (Park et al., 2019)."""

    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 16, num_masks: int = 2) -> None:
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def forward(self, spectrogram: Tensor) -> Tensor:
        augmented = spectrogram.clone()
        mask_value = float(spectrogram.mean())
        for _ in range(self.num_masks):
            max_freq_width = min(self.freq_mask_param, spectrogram.shape[-2])
            max_time_width = min(self.time_mask_param, spectrogram.shape[-1])
            freq_width = int(torch.randint(max_freq_width + 1, ()).item())
            time_width = int(torch.randint(max_time_width + 1, ()).item())
            if freq_width:
                freq_start = int(torch.randint(spectrogram.shape[-2] - freq_width + 1, ()).item())
                augmented[..., freq_start : freq_start + freq_width, :] = mask_value
            if time_width:
                time_start = int(torch.randint(spectrogram.shape[-1] - time_width + 1, ()).item())
                augmented[..., :, time_start : time_start + time_width] = mask_value
        return augmented


class AugmentedDataset(Dataset[tuple[Tensor, int]]):
    """Apply SpecAugment to cached, already normalized spectrograms."""

    def __init__(self, base_dataset: Dataset[tuple[Tensor, int]], transform: nn.Module | None = None) -> None:
        self.base_dataset = base_dataset
        self.transform = transform if transform is not None else SpecAugment()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        spectrogram, label = self.base_dataset[index]
        with torch.no_grad():
            spectrogram = self.transform(spectrogram)
        return spectrogram, label


class WaveformAugmentedDataset(Dataset[tuple[Tensor, int]]):
    """Create train-only waveform and spectrogram augmentation (Park et al., 2019)."""

    def __init__(
        self,
        dataset: MelSpectrogramDataset,
        indices: Sequence[int],
        noise_dir: str | Path | None = None,
        snr_min: float = 5.0,
        snr_max: float = 20.0,
        max_shift_ms: float = 100.0,
        p_augment: float = 1.0,
        freq_mask_param: int = 8,
        time_mask_param: int = 16,
        num_masks: int = 2,
        use_pitch_shift: bool = False,
        use_time_stretch: bool = False,
    ) -> None:
        if not 0.0 <= p_augment <= 1.0:
            raise ValueError("p_augment must be between 0 and 1")
        if snr_min > snr_max:
            raise ValueError("snr_min must be less than or equal to snr_max")
        if max_shift_ms < 0:
            raise ValueError("max_shift_ms must be non-negative")

        self.dataset = dataset
        self.indices = list(indices)
        self.sample_rate = int(dataset.metadata["sample_rate"])
        self.n_mels = int(dataset.metadata["n_mels"])
        self.mean = float(dataset.metadata["mean"])
        self.std = float(dataset.metadata["std"])
        self.data_dir = Path(str(dataset.metadata.get("data_dir", "")))
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.max_shift_samples = int(self.sample_rate * max_shift_ms / 1000.0)
        self.p_augment = p_augment
        self.use_pitch_shift = use_pitch_shift
        self.use_time_stretch = use_time_stretch
        self.mel_transform = _create_mel_transform(self.sample_rate, self.n_mels)
        self.spec_augment = SpecAugment(freq_mask_param, time_mask_param, num_masks)
        self.noise_waveforms = self._load_noise_waveforms(noise_dir)

    @staticmethod
    def _load_noise_waveforms(noise_dir: str | Path | None) -> list[Tensor]:
        if noise_dir is None:
            return []
        noise_path = Path(noise_dir)
        if not noise_path.exists():
            raise FileNotFoundError(f"Noise directory not found: {noise_path}")
        waveforms: list[Tensor] = []
        for path in sorted(noise_path.glob("*.wav")):
            waveform, sample_rate = _load_waveform(path)
            if sample_rate != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
            waveforms.append(waveform.mean(dim=0, keepdim=True))
        if not waveforms:
            raise ValueError(f"No noise .wav files found in {noise_path}")
        return waveforms

    def __len__(self) -> int:
        return len(self.indices)

    def _source_path(self, item_index: int) -> Path:
        item = self.dataset.items[item_index]
        wav_path = str(item.get("wav_path", ""))
        if not wav_path or not self.data_dir:
            raise RuntimeError(
                "The cache does not contain source wav paths. Rebuild it with --build-cache before using --augment-mode full."
            )
        source_path = self.data_dir / wav_path
        if not source_path.exists():
            raise FileNotFoundError(f"Source wav file from cache metadata not found: {source_path}")
        return source_path

    def _time_shift(self, waveform: Tensor) -> Tensor:
        if self.max_shift_samples == 0:
            return waveform
        shift = int(torch.randint(-self.max_shift_samples, self.max_shift_samples + 1, ()).item())
        shifted = torch.zeros_like(waveform)
        if shift >= 0:
            shifted[:, shift:] = waveform[:, : waveform.shape[1] - shift]
        else:
            shifted[:, :shift] = waveform[:, -shift:]
        return shifted

    def _noise_injection(self, waveform: Tensor) -> Tensor:
        if not self.noise_waveforms:
            return waveform
        noise = self.noise_waveforms[int(torch.randint(len(self.noise_waveforms), ()).item())]
        if noise.shape[1] < CLIP_SAMPLES:
            repeats = math.ceil(CLIP_SAMPLES / noise.shape[1])
            noise = noise.repeat(1, repeats)
        start = int(torch.randint(noise.shape[1] - CLIP_SAMPLES + 1, ()).item())
        noise = noise[:, start : start + CLIP_SAMPLES]
        signal_rms = waveform.pow(2).mean().sqrt()
        noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-8)
        snr_db = torch.empty(()).uniform_(self.snr_min, self.snr_max).item()
        scale = signal_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
        return waveform + noise * scale

    def _optional_waveform_effects(self, waveform: Tensor) -> Tensor:
        if self.use_pitch_shift:
            steps = float(torch.empty(()).uniform_(-1.0, 1.0).item())
            waveform = torchaudio.functional.pitch_shift(waveform, self.sample_rate, steps)
        if self.use_time_stretch:
            # Speed perturbation is used here as a lightweight waveform-domain alternative to phase vocoding.
            rate = float(torch.empty(()).uniform_(0.9, 1.1).item())
            resampled_rate = max(1, int(round(self.sample_rate * rate)))
            waveform = torchaudio.functional.resample(waveform, resampled_rate, self.sample_rate)
        return _fix_length(waveform)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        item_index = self.indices[index]
        item = self.dataset.items[item_index]
        label = int(item["label"])
        if torch.rand(()) > self.p_augment:
            return self.dataset[item_index]

        waveform, source_rate = _load_waveform(self._source_path(item_index))
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = _fix_length(waveform)
        waveform = self._time_shift(waveform)
        waveform = self._noise_injection(waveform)
        waveform = self._optional_waveform_effects(waveform)
        mel_spec = self.mel_transform(waveform)
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0
        )
        mel_spec = (mel_spec - self.mean) / self.std
        with torch.no_grad():
            mel_spec = self.spec_augment(mel_spec)
        return mel_spec, label
