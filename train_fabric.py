import wandb

from data import DummyDataset

from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer
from musiclm_pytorch.trainer_fabric import FabricTrainer

import torch

torch.manual_seed(42)
from lightning.fabric.utilities.rank_zero import rank_zero_only


class SimpleLogger:
    def __init__(self):
        import wandb

        wandb.init(project="MusicLMDummyData", name="Fabric")

    @rank_zero_only
    def __call__(self, logs: dict):
        wandb.log(logs)


audio_transformer = AudioSpectrogramTransformer(
    dim=512,
    depth=6,
    heads=8,
    dim_head=64,
    spec_n_fft=128,
    spec_win_length=24,
    spec_aug_stretch_factor=0.8,
)

text_transformer = TextTransformer(dim=512, depth=6, heads=8, dim_head=64)

mulan = MuLaN(audio_transformer=audio_transformer, text_transformer=text_transformer)

# get a ton of <sound, text> pairs and train
dataset = DummyDataset(
    num_samples=500000,
    sample_length_audio=1024,
    sample_length_text=256,
    num_classes=2000,
    seed=42,
)

trainer = FabricTrainer(
    mulan=mulan,
    dataset=dataset,
    num_train_steps=500000,
    batch_size=2,
)
trainer.train(SimpleLogger())
