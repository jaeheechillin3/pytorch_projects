# [Paper Review] Attention Is All You Need

> **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin\
> **Institutions:** Google Brain, Google Research, University of Toronto\
> **Publication:** NeurIPS 2017\
> **Link:** https://arxiv.org/abs/1706.03762

## Abstract & Introduction

Recurrent neural networks (RNNs) have long been the architecture of choice for natural language processing tasks such as machine translation. However, the sequential RNNs have two large shortcomings: parallelization is difficult, and relationships between far apart tokens are difficult to model. There have been attempts to use convolutional neural networks (CNNs) alongside RNNs in order to reduce sequential computation. 

This paper proposes the Transformer, a model that relies entirely on the attention mechanism to model global dependencies between input and output sequences. Without reliance on any recurrence or convolution, the model achieves significantly more parallelization and reaches state of the art results on machine translation benchmarks. 

## Methods

### Overall architecture

The base model has six encoder and six decoder layers, with $d_{\mathrm{model}}=512$ and eight attention heads. The encoder produces contextual source representations; the decoder predicts a target sequence conditioned on them. The encoder is bidirectional, while the decoder must respect the order in which words become available.

<p align="center">
  <img src="figure1.png" width="600">
</p>

<p align="center"><em>Figure 1. The Transformer model architecture.</em></p>


### Attention Mechanism 

<p align="center">
  <img src="figure1.png" width="600">
</p>

<p align="center"><em>Figure 1. The Transformer model architecture.</em></p>

<p align="center">
  <img src="figure1.png" width="600">
</p>

<p align="center"><em>Figure 1. The Transformer model architecture.</em></p>


Scaled dot-product attention computes

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

For one head, an input $X\in\mathbb R^{n\times d_{\mathrm{model}}}$ is projected into $Q=XW^Q$, $K=XW^K$, and $V=XW^V$. Then $QK^\top$ has shape $n\times n$: row $i$ contains the scores from token $i$ to every token. Softmax operates across each row, and multiplication by $V$ produces a weighted combination of value vectors for each query.

Queries select relevant keys; the resulting weights combine values. Scaling prevents large dot products from saturating softmax. Multi-head attention concatenates independently projected attention outputs.

The scale factor can be motivated by assuming independent query and key components with mean zero and variance one. Their dot product sums $d_k$ terms and has variance $d_k$; dividing by $\sqrt{d_k}$ keeps its variance near one. Without scaling, softmax can become extremely concentrated, leaving small gradients.

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

### Training 

The model was trained on the WMT 2014 English-German dataset and the WMT 2014 English-French dataset, containing 4.5 million and 36 million sentences each. The Adam optimizer, residual dropout and label smoothing was used. The model was trained on eight NVIDIA P100 GPUs. 

## Results

### Translation quality and efficiency

The Transformer achieves state-of-the-art BLEU scores on both English-to-German and English-to-French translation tasks as the fraction of the training cost. 

<p align="center">
  <img src="images/table4.jpeg" width="600">
</p>

<p align="center"><em>Table 2. BLUE scores and training costs of English-to-German and English-to-French translation models.</em></p>

### English Constituency Parsing

In order to evaluate the Transformer for other NLP tasks, its performance on English constituency parsing was experimented. The Wall Street Journal (WSJ) portion of the Penn Treebank and BerkeleyParser datasets were used. Despite the lack of task-specific tuning, the Transformer outperformed all previous models except one. 

<p align="center">
  <img src="images/table4.jpeg" width="600">
</p>

<p align="center"><em>Table 4. English constituency parsing results. </em></p>

