<img src="./musiclm.png" width="450px"></img>

## MusicLM - Pytorch (wip)

Implementation of <a href="https://google-research.github.io/seanet/musiclm/examples/">MusicLM</a>, Google's new SOTA model for music generation using attention networks, in Pytorch.

They are basically using text-conditioned <a href="https://github.com/lucidrains/audiolm-pytorch">AudioLM</a>, but surprisingly with the embeddings from a text-audio contrastive learned model named <a href="https://arxiv.org/abs/2208.12415">MuLan</a>. MuLan is what will be built out in this repository, with AudioLM modified from the other repository to support the music generation needs here.

## Usage

```install
$ pip install musiclm-pytorch
```

## Usage

`MuLaN` first needs to be trained

```python
import torch
from musiclm_pytorch import MuLaN, AudioSpectrogramTransformer, TextTransformer

audio_transformer = AudioSpectrogramTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    spec_n_fft = 128,
    spec_win_length = 24,
    spec_aug_stretch_factor = 0.8
)

text_transformer = TextTransformer(
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64
)

mulan = MuLaN(
    audio_transformer = audio_transformer,
    text_transformer = text_transformer
)

texts = torch.randint(0, 20000, (2, 256))
wavs = torch.randn(2, 1024)

loss = mulan(wavs, texts)
loss.backward()
```

## Todo

- [ ] wrap mulan with mulan wrapper and quantize the output, project to audiolm dimensions
- [ ] modify audiolm to accept conditioning embeddings, optionally take care of different dimensions through a separate projection
- [ ] audiolm and mulan goes into musiclm and generate, filter with mulan

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work and open source cutting edge artificial intelligence research

## Citations

```bibtex
@inproceedings{Agostinelli2023MusicLMGM,
    title     = {MusicLM: Generating Music From Text},
    author    = {Andrea Agostinelli and Timo I. Denk and Zal{\'a}n Borsos and Jesse Engel and Mauro Verzetti and Antoine Caillon and Qingqing Huang and Aren Jansen and Adam Roberts and Marco Tagliasacchi and Matthew Sharifi and Neil Zeghidour and C. Frank},
    year      = {2023}
}
```

```bibtex
@article{Huang2022MuLanAJ,
    title   = {MuLan: A Joint Embedding of Music Audio and Natural Language},
    author  = {Qingqing Huang and Aren Jansen and Joonseok Lee and Ravi Ganti and Judith Yue Li and Daniel P. W. Ellis},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.12415}
}
```

*The only truth is music.* - Jack Kerouac
