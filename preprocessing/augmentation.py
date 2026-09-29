from __future__ import annotations

import torch
import torchaudio
from torch import Tensor, nn
from torch.utils.data import Dataset


class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 16, num_masks: int = 2) -> None:
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.num_masks = num_masks

    def forward(self, spectrogram: Tensor) -> Tensor:
        augmented = spectrogram.clone()
        for _ in range(self.num_masks):
            augmented = self.freq_mask(augmented)
            augmented = self.time_mask(augmented)
        return augmented


class AugmentedDataset(Dataset[tuple[Tensor, int]]):
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
