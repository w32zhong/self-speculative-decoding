# Self-Speculative Decoding

Code associated with the paper:

**[Draft &amp; Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)**

![Overview](./assets/intro.pdf)

Self-Speculative Decoding is a novel inference scheme for accelerating Large Language Models (LLMs) without additional neural network training and extra memory footprint. It not only maintains consistent output quality but also ensures model compatibility, making it a *plug-and-play* and *cost-effective* solution for LLM inference acceleration.

Self-Speculative Decoding involves a two-stage process:

**Drafting stage:** Generates draft tokens by selectively skipping certain intermediate layers.

**Verification stage:** Employs the original LLM to validate draft tokens in one forward pass.

## Cite Our Paper

If you find this code useful in your research, please consider citing:

```
@article{zhang2023draft,
      title={Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding}, 
      author={Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, Sharad Mehrotra},
      year={2023},
      eprint={2309.08168},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Coming Soon

Code under construction.
