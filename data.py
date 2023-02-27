from typing import Tuple

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 5000,
        sample_length_audio: int = 1024,
        sample_length_text: int = 256,
        num_classes: int = 20000,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.num_samples = num_samples
        self.sample_length_audio = sample_length_audio
        self.sample_length_text = sample_length_text
        self.num_classes = num_classes

        generator = torch.Generator().manual_seed(seed)

        self.dummy_audio = torch.rand(
            (self.num_samples, self.sample_length_audio), generator=generator
        )
        self.dummy_text = torch.randint(
            0,
            self.num_classes,
            (self.num_samples, self.sample_length_text),
            generator=generator,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.dummy_audio[idx],
            self.dummy_text[idx],
        )

    def __len__(self) -> int:
        return self.num_samples
