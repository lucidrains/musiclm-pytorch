from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum

from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking

from audiolm_pytorch import AudioLM
from audiolm_pytorch.utils import AudioConditionerBase

from x_clip.tokenizer import tokenizer
from vector_quantize_pytorch import ResidualVQ

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from beartype.typing import List, Optional, Tuple
from beartype import beartype

# functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# decorators

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# tensor functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)

    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)

# biasless layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        b, n, _, device = *x.shape, x.device

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # split for multi-headed attention

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # qk rmsnorm, technique circulating within brain used to stabilize a 22B parameter vision model training

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):

        for attn, ff in self.layers:
            x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x
            x = ff(x) + x

        return x

# Audio Spectrogram Transformer - https://arxiv.org/abs/2104.01778

def pair(t):
    return (t, t) if not isinstance(t, tuple) else t

class AudioSpectrogramTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        patch_size = 16,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        spec_n_fft = 128,
        spec_power = 2,
        spec_win_length = 24,
        spec_hop_length = None,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 80,
        spec_aug_time_mask = 80,
        patch_dropout_prob = 0.25
    ):
        super().__init__()
        self.dim = dim

        self.patch_size = pair(patch_size)
        patch_input_dim = self.patch_size[0] * self.patch_size[1]

        self.to_patch_tokens = Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = self.patch_size[0], p2 = self.patch_size[1]),
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, dim),
            nn.LayerNorm(dim)
        )

        self.spec = Spectrogram(
            n_fft = spec_n_fft,
            power = spec_power,
            win_length = spec_win_length,
            hop_length = spec_hop_length,
            pad = spec_pad,
            center = spec_center,
            pad_mode = spec_pad_mode
        )

        # SpecAugment - seems to be widely used in audio field https://arxiv.org/abs/1904.08779

        self.aug = torch.nn.Sequential(
            TimeStretch(spec_aug_stretch_factor, fixed_rate = True),
            FrequencyMasking(freq_mask_param = spec_aug_freq_mask),
            TimeMasking(time_mask_param = spec_aug_time_mask),
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout
        )

        self.norm = LayerNorm(dim)

        # patch dropout

        self.patch_dropout_prob = patch_dropout_prob

        # 2d dynamic positional bias

        mlp_hidden_dim = dim // 4

        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    def forward(
        self,
        x,
        force_no_patch_dropout = False
    ):
        batch, device = x.shape[0], x.device

        x = self.spec(x)

        if self.training:
            x = self.aug(x)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes

        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size

        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): # just keep printing to be annoying until it is fixed
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        x = x[..., :rounded_height, :rounded_width]

        # to patches

        x = self.to_patch_tokens(x)

        # get number of patches along height and width

        _, num_patch_height, num_patch_width, _ = x.shape

        # get 2d relative positions

        grid = torch.stack(torch.meshgrid(
            torch.arange(num_patch_height, device = device),
            torch.arange(num_patch_width, device = device)
        , indexing = 'ij'), dim = -1)

        grid = rearrange(grid, '... c -> (...) c')

        # 2d sinusoidal positional embedding

        x = x + posemb_sincos_2d(x)

        x = rearrange(x, 'b ... c -> b (...) c')

        # patch dropout

        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            n, device = x.shape[1], x.device

            batch_indices = torch.arange(batch, device = device)
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device = device).topk(num_patches_keep, dim = -1).indices

            x = x[batch_indices, patch_indices_keep]

            grid = repeat(grid, '... -> b ...', b = batch)
            grid = grid[batch_indices, patch_indices_keep]

        # 2d relative positional bias

        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())

        # attention, what else

        x = self.transformer(x, rel_pos_bias = rel_pos_bias)

        # final global average and norm (most recent papers show this is superior to CLS token)

        x = reduce(x, 'b n d -> b d', 'mean')

        return self.norm(x)

# text transformer

@beartype
class TextTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens = tokenizer.vocab_size,
        max_seq_len = 256,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        pad_id = 0
    ):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len

        self.cls_token = nn.Parameter(torch.randn(dim))

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.pad_id = pad_id
        self.norm = LayerNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x = None,
        raw_texts: Optional[List[str]] = None,
        mask = None
    ):
        assert exists(x) ^ exists(raw_texts)

        if exists(raw_texts):
            x = tokenizer.tokenize(raw_texts).to(self.device)

        if not exists(mask):
            mask = x != self.pad_id

        b, n, device = *x.shape, x.device

        # token embedding + positional embedding

        x = self.token_emb(x)

        assert n <= self.max_seq_len, f'text sequence length {n} must be less than {self.max_seq_len}'

        x = x + self.pos_emb(torch.arange(n, device = device))

        # cls tokens, as in bert

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')

        # account for attending to cls token with self attention mask

        mask = F.pad(mask, (1, 0), value = True)

        # attention

        x = self.transformer(x, mask = mask)

        # unpack the cls tokens

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.norm(cls_tokens)

# main classes

@beartype
class MuLaN(nn.Module):
    def __init__(
        self,
        audio_transformer: AudioSpectrogramTransformer,
        text_transformer: TextTransformer,
        dim_latent = 128,                       # they use 128
        decoupled_contrastive_learning = True,  # think this was used, make it optional
    ):
        super().__init__()
        self.dim_latent = dim_latent

        self.audio = audio_transformer
        self.text = text_transformer

        self.temperature = nn.Parameter(torch.tensor(1.))

        self.text_to_latents = nn.Linear(self.text.dim, dim_latent)
        self.audio_to_latents = nn.Linear(self.audio.dim, dim_latent)

        self.decoupled_contrastive_learning = decoupled_contrastive_learning

    def get_audio_latents(
        self,
        wavs
    ):
        audio_embeds = self.audio(wavs)
        audio_latents = self.audio_to_latents(audio_embeds)
        return l2norm(audio_latents)

    def get_text_latents(
        self,
        texts = None,
        raw_texts: Optional[List[str]] = None
    ):
        text_embeds = self.text(texts, raw_texts = raw_texts)
        text_latents = self.text_to_latents(text_embeds)
        return l2norm(text_latents)

    def forward(
        self,
        wavs,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_similarities = False
    ):
        batch, device = wavs.shape[0], wavs.device

        audio_latents = self.get_audio_latents(wavs)
        text_latents = self.get_text_latents(texts, raw_texts = raw_texts)

        cosine_sim = einsum('i d, j d -> i j', audio_latents, text_latents)

        assert cosine_sim.shape[0] == cosine_sim.shape[1], 'batch sizes for audio and text are not equal'

        if return_similarities:
            return cosine_sim

        cosine_sim = cosine_sim * self.temperature.exp()

        cosine_sim_exp = cosine_sim.exp()

        numerator = cosine_sim_exp.diag()

        if self.decoupled_contrastive_learning:
            eye = torch.eye(batch, device = device, dtype = torch.bool)
            cosine_sim_exp = cosine_sim_exp.masked_fill(eye, 0.)

        denominator = reduce(cosine_sim_exp, 'i j -> i', 'sum')

        contrastive_loss = -log(numerator) + log(denominator)
        return contrastive_loss.mean()

# music lm

@beartype
class MuLaNEmbedQuantizer(AudioConditionerBase):
    def __init__(
        self,
        mulan: MuLaN,
        conditioning_dims: Tuple[int, ...],
        rq_num_quantizers = 8,
        rq_ema_decay = 0.9,
        codebook_size = 1024,
        namespaces: Tuple[str, ...] = ('semantic', 'coarse', 'fine'),

    ):
        super().__init__()
        self.mulan = mulan

        assert len(namespaces) > 0
        self.namespaces = namespaces
        self.conditioning_dims = conditioning_dims

        assert len(conditioning_dims) == len(namespaces), 'number of conditioning dimensions must be equal to number of namespaces'

        dim = mulan.dim_latent

        self.rq = ResidualVQ(
            dim = dim,
            num_quantizers = rq_num_quantizers,
            codebook_size = codebook_size,
            decay = rq_ema_decay,
            commitment_weight = 0,    # only use EMA to update codebooks
            kmeans_init = True,
            threshold_ema_dead_code = 2,
            quantize_dropout = False  # no quantize dropout
        )

        self.dim = dim
        self.num_codebooks = rq_num_quantizers

        self.cond_embeddings = nn.ParameterDict({})

        for namespace, conditioning_dim in zip(namespaces, conditioning_dims):
            cond_embeddings = nn.Parameter(torch.randn(rq_num_quantizers, codebook_size, conditioning_dim))
            nn.init.normal_(cond_embeddings, std = 0.02)

            self.cond_embeddings[namespace] = cond_embeddings

        self.set_default_namespace(namespaces[0])

    def parameters(self):
        return self.cond_embeddings.parameters()

    def set_default_namespace(self, namespace):
        self._default_namespace = namespace

    def forward(
        self,
        wavs = None,
        texts = None,
        namespace = None
    ):
        assert exists(wavs) ^ exists(texts)

        namespace = default(namespace, self._default_namespace)
        assert namespace in self.namespaces, f'namespace {namespace} not found'
        cond_embeddings = self.cond_embeddings[namespace]

        with torch.no_grad():
            self.mulan.eval()

            # sound and language live in joint embedding space because of contrastive learning

            if exists(wavs):
                latents = self.mulan.get_audio_latents(wavs)
            elif exists(texts):
                latents = self.mulan.get_text_latents(texts)

        _, indices, _ = self.rq(latents)

        batch, num_codebooks, dim = indices.shape[0], self.num_codebooks, cond_embeddings.shape[-1]

        cond_embeddings = repeat(cond_embeddings, 'q c d -> b q c d', b = batch)
        indices = repeat(indices, 'b q -> b q 1 d', q = num_codebooks, d = dim)

        cond_embeddings = cond_embeddings.gather(2, indices)
        return rearrange(cond_embeddings, 'b q 1 d -> b q d')

@beartype
class MusicLM(nn.Module):
    def __init__(
        self,
        audio_lm: AudioLM,
        mulan_embed_quantizer: MuLaNEmbedQuantizer
    ):
        super().__init__()
        assert not exists(audio_lm.audio_conditioner), 'mulan must not have been passed into AudioLM. it will be managed externally now, embedding the text into the joint embedding space for text-to-audio synthesis'

        self.mulan_embed_quantizer = mulan_embed_quantizer
        self.audio_lm = audio_lm

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        raw_texts: List[str],
        **audio_lm_kwargs
    ):
        self.eval()

        texts = tokenizer.tokenize(raw_texts).to(self.device)

        text_embeds = self.mulan_embed_quantizer(texts = texts)

        return self.audio_lm(text_embeds = text_embeds, **audio_lm_kwargs)
