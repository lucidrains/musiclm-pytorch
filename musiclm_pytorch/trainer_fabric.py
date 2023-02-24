import shutil
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Optional, Tuple, Callable, Union

import torch
from beartype.door import is_bearable
from lightning.fabric import Fabric
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from musiclm_pytorch import MuLaN
from itertools import cycle


class FabricTrainer:
    def __init__(
        self,
        mulan: MuLaN,
        dataset: Dataset,
        batch_size: int,
        num_train_steps: Optional[int] = None,
        data_max_length: Optional[int] = None,
        lr: float = 3e-4,
        grad_accum_every: int = 1,
        betas: Tuple[float, float] = (0.9, 0.99),
        max_grad_norm: float = 0.5,
        valid_frac: float = 0.05,
        random_split_seed: int = 42,
        save_model_every: int = 1000,
        results_folder: str = "./results",
        remove_previous_results: bool = True,
        **fabric_kwargs,
    ):
        if batch_size <= 1:
            raise ValueError(
                "batch_size must be greater than 1 for constrastive learning (but ideally as large as possible)"
            )
        self.fabric = Fabric(**fabric_kwargs)

        self.mulan = mulan
        self.current_step = 0
        self.num_training_steps = num_train_steps or len(dataset)
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers
        self.optim = Adam(mulan.parameters(), lr=lr, betas=betas)

        self.max_grad_norm = max_grad_norm
        self.data_max_length = data_max_length

        # create dataset
        self.ds = dataset
        self.ds_fields = None

        # split for validation
        if valid_frac > 0:
            train_size = int((1 - valid_frac)) * len(self.ds)
            valid_size = len(self.ds) - train_size
            self.train_ds, self.valid_ds = random_split(
                self.ds,
                [train_size, valid_size],
                generator=torch.Generator().manual_seed(random_split_seed),
            )
            self.fabric.print(
                f"Training with dataset of {len(self.train_ds)} samples and "
                "validating with randomly selected {len(self.valid_ds)} samples."
            )
        else:
            self.valid_ds = self.train_ds = self.ds
            self.fabric.print(
                f"Training with shared training and validation dataset of {len(self.ds)} samples. "
                "This may not reflect real world performance on unseen data!"
            )

        self.train_dl = DataLoader(self.train_ds, batch_size=batch_size, collate_fn=_curtail_to_shortest_collate, shuffle=True, drop_last=True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=batch_size, collate_fn=_curtail_to_shortest_collate, shuffle=True, drop_last=True)

        self.mulan, self.optim = self.fabric.setup(self.mulan, self.optim)
        self.train_dl, self.valid_dl = self.fabric.setup_dataloaders(self.train_dl, self.valid_dl)

        self.train_iter = cycle(self.train_dl)
        self.valid_iter = cycle(self.valid_dl)
        self.save_model_every = save_model_every

        self.results_folder = Path(results_folder)

        if len(list(self.results_folder.rglob('*'))) and remove_previous_results:
            shutil.rmtree((str(self.results_folder)))

        self.results_folder.mkdir(parents=True, exist_ok=True)

    @property
    def state(self):
        return {'model': self.mulan, 'optimizer': self.optim, 'current_step': self.current_step}

    def save(self):
        self.fabric.save(self.results_folder / f'mulan-step-{self.current_step:05d}.ckpt', self.state)

    def load(self, path: Optional[Union[str, Path]] = 'latest'):

        if path == 'latest':
            avail_files = list(self.results_folder.glob('*.ckpt'))
            if not avail_files:
                raise RuntimeError(f'There are no files to load from in {str(self.results_folder)}. Please specify the correct path.')

            path = sorted(avail_files, key=lambda x: int(str(x).rsplit('-', 1)[-1].rsplit('.', 1)[0]))[-1]

        path = Path(path)
        assert path.is_file()

        remaining_items = self.fabric.load(path, self.state)

        self.current_step = remaining_items.pop('current_step')

    def train(self, log_fn: Optional[Callable] = None):
        while self.current_step < self.num_training_steps:
            logs = defaultdict(lambda: 0.)

            self.mulan.train()

            for _ in range(self.grad_accum_every):
                data_kwargs =  _data_tuple_to_kwargs(next(self.train_iter), self.data_max_length)

                loss = self.mulan(**data_kwargs)
                averaged_loss = loss / self.grad_accum_every
                self.fabric.backward(averaged_loss)

                logs['loss'] += averaged_loss.item()

            if self.max_grad_norm:
                scaler = getattr(self.fabric.precision, 'scaler', None)
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.clip_grad_norm_(self.mulan.parameters(), self.max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

            self.fabric.print(f"{steps}: loss: {logs['loss']}")
            self.fabric.log('train_loss', logs['loss'])
            if log_fn is not None:
                log_fn(logs)


def _collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = torch.stack(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner


@_collate_one_or_multiple_tensors
def _curtail_to_shortest_collate(data):
    min_len = min([datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)


@_collate_one_or_multiple_tensors
def _pad_to_longest(data):
    return pad_sequence(data, batch_first=True)

def _data_tuple_to_kwargs(data, data_max_length):
    ds_fields = _determine_types(data, DATASET_FIELD_TYPE_CONFIG)
    data_kwargs = dict(zip(ds_fields, data))

    wavs = data_kwargs['wavs']
    data_kwargs.update(wavs=wavs[..., :data_max_length])
    return data_kwargs


def _determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)
