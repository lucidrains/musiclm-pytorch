<img src="./musiclm.png" width="450px"></img>

## MusicLM - Pytorch (wip)

Implementation of <a href="https://google-research.github.io/seanet/musiclm/examples/">MusicLM</a>, Google's new SOTA model for music generation using attention networks, in Pytorch.

They are basically using text-conditioned <a href="https://github.com/lucidrains/audiolm-pytorch">AudioLM</a>, but surprisingly with the embeddings from a text-audio contrastive learned model named <a href="https://arxiv.org/abs/2208.12415">MuLan</a>. MuLan is what will be built out in this repository, with AudioLM modified from the other repository to support the music generation needs here.

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work and open source cutting edge artificial intelligence research

## Citations

```bibtex
@article{Mittal2021SymbolicMG,
    title   = {Symbolic Music Generation with Diffusion Models},
    author  = {Gautam Mittal and Jesse Engel and Curtis Hawthorne and Ian Simon},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2103.16091}
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
