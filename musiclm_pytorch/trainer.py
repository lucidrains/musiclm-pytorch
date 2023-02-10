import copy
from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import wraps, partial

from typing_extensions import Annotated

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is
from beartype.typing import Union, List, Optional, Tuple, Callable

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from musiclm_pytorch import MuLaN

from einops import rearrange

from accelerate import Accelerator

# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
    wavs = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
    raw_texts = List[str],
    texts = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.long and t.ndim == 2]
    ],
)

# helpers

def exists(val):
    return val is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)

# optimizer functions

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params

# dataloader functions

def collate_one_or_multiple_tensors(fn):
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

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)

# semantic transformer trainer

@beartype
class MuLaNTrainer(nn.Module):
    def __init__(
        self,
        mulan: MuLaN,
        dataset: Dataset,
        *,
        num_train_steps = None,
        batch_size,
        data_max_length = None,
        folder = None,
        lr = 3e-4,
        grad_accum_every = 1,
        betas = (0.9, 0.99),
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        force_clear_prev_results = None  # set to True | False to skip the prompt
    ):
        super().__init__()
        assert batch_size > 1, 'batch size must be greater than 1 for contrastive learning (but ideally as large as possible)'

        self.accelerator = Accelerator(**accelerate_kwargs)

        self.mulan = mulan

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = default(num_train_steps, len(dataset)) # 1 epoch by default
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = Adam(mulan.parameters(), lr = lr, betas = betas)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        self.data_max_length = data_max_length

        # create dataset

        self.ds = dataset
        self.ds_fields = None

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, pad_to_longest = False, drop_last = True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, pad_to_longest = False, drop_last = True)

        # prepare with accelerator

        (
            self.mulan,
            self.optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.mulan,
            self.optim,
            self.dl,
            self.valid_dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every

        hps = dict(
            num_train_steps = num_train_steps,
            data_max_length = data_max_length,
            learning_rate = lr
        )

        self.accelerator.init_trackers("mulan", config = hps)

        # results folder

        self.results_folder = Path(results_folder)

        if force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        # to device

        self.mulan.to(self.device)

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.mulan),
            optim = self.optim.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path))

        mulan = self.accelerator.unwrap_model(self.mulan)
        mulan.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def data_tuple_to_kwargs(self, data):
        if not exists(self.ds_fields):
            self.ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'

        data_kwargs =  dict(zip(self.ds_fields, data))

        wavs = data_kwargs['wavs']
        data_kwargs.update(wavs = wavs[..., :self.data_max_length])

        return data_kwargs

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.mulan.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            data_kwargs = self.data_tuple_to_kwargs(next(self.dl_iter))

            loss = self.mulan(**data_kwargs)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.mulan.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step = steps)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.mulan.state_dict()
            model_path = str(self.results_folder / f'mulan.{steps}.pt')
            torch.save(state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn: Callable = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
