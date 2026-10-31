# [Paper Review] Attention Is All You Need

> **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin\
> **Institutions:** Google Brain, Google Research, University of Toronto\
> **Publication:** NeurIPS 2017\
> **Link:** https://arxiv.org/abs/1706.03762

## Abstract & Introduction

Recurrent sequence models process positions sequentially, limiting training parallelism. The Transformer replaces recurrence and convolution with attention, allowing direct interaction between distant tokens.

Bahdanau attention already lets a decoder consult different source positions. The Transformer extends the use of attention to building the source and target representations themselves. The important shift is that tokens can exchange information directly, rather than first passing it through a chain of recurrent states.

## Methods

### Overall architecture

The base model has six encoder and six decoder layers, with $d_{\mathrm{model}}=512$ and eight attention heads. The encoder produces contextual source representations; the decoder predicts a target sequence conditioned on them. The encoder is bidirectional, while the decoder must respect the order in which words become available.

### Scaled dot-product attention

Scaled dot-product attention computes

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

For one head, an input $X\in\mathbb R^{n\times d_{\mathrm{model}}}$ is projected into $Q=XW^Q$, $K=XW^K$, and $V=XW^V$. Then $QK^\top$ has shape $n\times n$: row $i$ contains the scores from token $i$ to every token. Softmax operates across each row, and multiplication by $V$ produces a weighted combination of value vectors for each query.

Queries select relevant keys; the resulting weights combine values. Scaling prevents large dot products from saturating softmax. Multi-head attention concatenates independently projected attention outputs.

The scale factor can be motivated by assuming independent query and key components with mean zero and variance one. Their dot product sums $d_k$ terms and has variance $d_k$; dividing by $\sqrt{d_k}$ keeps its variance near one. Without scaling, softmax can become extremely concentrated, leaving small gradients.

### Multi-head attention

For head $r$,

$
\mathrm{head}_r=\operatorname{Attention}(QW_r^Q,KW_r^K,VW_r^V),
$

$
\operatorname{MultiHead}(Q,K,V)
=\operatorname{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O.
$

Here $Q,K,V$ denote the inputs before the head-specific projections. In the base model, each of eight heads uses 64-dimensional keys and values. Concatenating their outputs restores 512 dimensions, and $W^O$ combines the heads. Different projections allow several patterns of interaction at once; individual heads are not explicitly assigned grammatical roles.

### Self-attention, cross-attention, and masking

In self-attention, $Q$, $K$, and $V$ are projections of the same sequence. In cross-attention, queries come from the decoder and keys and values from the encoder. Causal masking sets future-position scores to $-\infty$ before softmax, preventing access to later target words.

A decoder position therefore combines two kinds of information: what has already been written, through masked self-attention, and what the source says, through cross-attention. Cross-attention can form a rectangular score matrix when source and target lengths differ. Padding masks separately prevent attention to artificial tokens added for batching.

### Feed-forward blocks and residual connections

Encoder layers combine self-attention and position-wise feed-forward networks. Decoder layers add causal masking and cross-attention to encoder outputs. Residual connections and layer normalization stabilize training; sinusoidal position encodings supply order information.

The feed-forward network applies the same two-layer transformation independently to each position, expanding from 512 to 2048 dimensions and back. Unlike attention, it mixes features within a token rather than information across tokens.

The original paper applies normalization after the residual addition:

$
\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).
$

This is the original *post-norm* arrangement. The residual path carries the previous representation forward while the sublayer learns an update. Layer normalization operates on each token's feature vector rather than mixing token positions.

### Position information

Self-attention alone does not encode token order. Permuting input tokens permutes its outputs, so the model needs position information to distinguish sentences containing the same words in different orders. The paper adds fixed sine and cosine vectors to token embeddings:

$
PE_{(p,2j)}=\sin\left(p/10000^{2j/d_{\mathrm{model}}}\right),\qquad
PE_{(p,2j+1)}=\cos\left(p/10000^{2j/d_{\mathrm{model}}}\right).
$

Different frequencies represent position at different scales. These encodings supply order information without requiring recurrent computation.

### Training versus generation

Training uses Adam with learning-rate warmup, dropout, and label smoothing. Target positions are processed in parallel during training, while generation remains autoregressive.

During training, the decoder receives the correct target sequence shifted by one position. For a target “I am here,” its input begins with a start token followed by “I am,” and its prediction targets are “I am here.” Causal masking prevents later input positions from revealing an answer. This makes all next-token losses computable in one forward pass without removing the autoregressive dependency.

At inference, the next word is unknown: the decoder predicts it, appends it to the prefix, and repeats. Thus, parallel training does not imply parallel generation of a complete translation.

## Results

### Translation quality and efficiency

The big model achieves **28.4 BLEU on WMT 2014 English–German** and **41.8 on English–French** (Table 2). The French result uses 3.5 days on eight P100 GPUs. Parsing experiments demonstrate transfer beyond translation.

These results concern the full encoder–decoder system, including its optimization and decoding choices. They demonstrate that recurrent computation is unnecessary for strong translation performance; they do not isolate attention as the only reason for every improvement.

### Ablations

The ablation study examines head count, attention dimensions, model size, and positional encodings. Learned and sinusoidal positions perform similarly, while a single attention head performs worse than the best multi-head configuration.

This supports the usefulness of multiple learned attention subspaces.

### Strengths and limitations

Self-attention shortens dependency paths, but its attention matrix grows quadratically with sequence length. For length $n$, every query compares with $n$ keys, yielding $n^2$ attention weights per head. Direct interaction makes distant tokens easier to connect, but long inputs can be costly in memory and computation. The original architecture also retains a sequential decoding loop.

The paper establishes attention as the central sequence-modeling operation.

For implementation detail, the supplementary [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) walks through the model in PyTorch; its code conventions should be distinguished from the original paper's exact architecture.
