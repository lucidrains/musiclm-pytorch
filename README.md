<img src="./musiclm.png" width="450px"></img>

## MusicLM - Pytorch (wip)

Implementation of <a href="https://google-research.github.io/seanet/musiclm/examples/">MusicLM</a>, Google's new SOTA model for music generation using attention networks, in Pytorch.

They are basically using text-conditioned <a href="https://github.com/lucidrains/audiolm-pytorch">AudioLM</a>, but surprisingly with the embeddings from a new text-audio contrastive learned model, which they named MuLan. MuLan is what will be built out in this repository, with AudioLM modified from the other repository to support the music generation needs here.

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
