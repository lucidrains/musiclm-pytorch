import torch
from torch import nn
from torch.autograd import Function
import torch.distributed as distributed

from einops import rearrange

# distributed helpers

def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, distributed.get_rank(), distributed.get_world_size()

    if not exists(sizes):
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
        distributed.all_gather(sizes, size)
        sizes = torch.stack(sizes)

    max_size = sizes.amax().item()
    padded_t = pad_dim_to(t, max_size, dim = dim)

    gathered_tensors = [torch.empty(padded_t.shape, device = device, dtype = padded_t.dtype) for i in range(world_size)]
    distributed.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    seq = torch.arange(max_size, device = device)

    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    mask = rearrange(mask, 'i j -> (i j)')
    seq = torch.arange(mask.shape[-1], device = device)
    indices = seq[mask]

    gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

class AllGatherFunction(Function):
    @staticmethod
    def forward(ctx, x, dim, sizes, all_reduce_grads):
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        ctx.dim = dim
        ctx.all_reduce_grads = all_reduce_grads
        ctx.batch_sizes = batch_sizes.tolist()
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        batch_sizes, rank = ctx.batch_sizes, distributed.get_rank()
        if ctx.all_reduce_grads:
            distributed.all_reduce(grads)

        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        return grads_by_rank[rank], None, None, None

class AllGather(nn.Module):
    def __init__(
        self,
        dim,
        *,
        all_reduce_grads = False
    ):
        super().__init__()
        self.dim = dim
        self.all_reduce_grads = all_reduce_grads
        self.is_distributed = distributed.is_initialized() and distributed.get_world_size() > 1

    def forward(
        self,
        x,
        sizes = None
    ):
        if not self.is_distributed:
            return x, None

        return AllGatherFunction.apply(x, self.dim, sizes, self.all_reduce_grads)
