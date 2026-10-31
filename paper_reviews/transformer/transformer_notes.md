# [Paper Review] Attention Is All You Need

> **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin\
> **Institutions:** Google Brain, Google Research, University of Toronto\
> **Publication:** NeurIPS 2017\
> **Link:** https://arxiv.org/abs/1706.03762

## Abstract & Introduction

Recurrent sequence models process positions sequentially, limiting training parallelism. The Transformer replaces recurrence and convolution with attention, allowing direct interaction between distant tokens.

## Methods

The base model has six encoder and six decoder layers, with $d_{\mathrm{model}}=512$ and eight attention heads. Scaled dot-product attention computes

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Queries select relevant keys; the resulting weights combine values. Scaling prevents large dot products from saturating softmax. Multi-head attention concatenates independently projected attention outputs.

In self-attention, $Q$, $K$, and $V$ are projections of the same sequence. In cross-attention, queries come from the decoder and keys and values from the encoder. Causal masking sets future-position scores to $-\infty$ before softmax, preventing access to later target words.

Encoder layers combine self-attention and position-wise feed-forward networks. Decoder layers add causal masking and cross-attention to encoder outputs. Residual connections and layer normalization stabilize training; sinusoidal position encodings supply order information.

The feed-forward network applies the same two-layer transformation independently to each position, expanding from 512 to 2048 dimensions and back. Unlike attention, it mixes features within a token rather than information across tokens.

Training uses Adam with learning-rate warmup, dropout, and label smoothing. Target positions are processed in parallel during training, while generation remains autoregressive.

## Results

The big model achieves **28.4 BLEU on WMT 2014 English–German** and **41.8 on English–French** (Table 2). The French result uses 3.5 days on eight P100 GPUs. Parsing experiments demonstrate transfer beyond translation.

The ablation study examines head count, attention dimensions, model size, and positional encodings. Learned and sinusoidal positions perform similarly, while a single attention head performs worse than the best multi-head configuration.

This supports the usefulness of multiple learned attention subspaces.

Self-attention shortens dependency paths, but its attention matrix grows quadratically with sequence length. The paper establishes attention as the central sequence-modeling operation.
