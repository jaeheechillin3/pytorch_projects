# [Paper Review] Attention Is All You Need

> **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin\
> **Institutions:** Google Brain, Google Research, University of Toronto\
> **Publication:** NeurIPS 2017\
> **Link:** https://arxiv.org/abs/1706.03762

## Abstract & Introduction

Recurrent neural networks (RNNs) have long been the architecture of choice for natural language processing tasks such as machine translation. However, the sequential RNNs have two large shortcomings: parallelization is difficult, and relationships between far apart tokens are difficult to model. There have been attempts to use convolutional neural networks (CNNs) alongside RNNs in order to reduce sequential computation. 

This paper proposes the Transformer, a model that relies entirely on the attention mechanism to model global dependencies between input and output sequences. Without reliance on any recurrence or convolution, the model achieves significantly more parallelization and reaches state of the art results on machine translation benchmarks. 

## Methods

### Model architecture

The Transformer uses an encoder to represent the source sequence and a decoder to generate target tokens conditioned on those representations and earlier target tokens.

<p align="center">
  <img src="./images/figure1.png" alt="Transformer encoder-decoder architecture" width="500">
</p>

*Figure 1. Transformer architecture (Vaswani et al., Figure 1).*

### 3.1 Encoder and decoder stacks

The base model has six encoder and six decoder layers, with $`d_{\mathrm{model}}=512`$. Each encoder layer contains multi-head self-attention and a position-wise feed-forward network. Each decoder layer contains masked self-attention, encoder-decoder attention, and a feed-forward network, in that order.

Every sublayer uses a residual connection followed by layer normalization:

```math
\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)).
```

All sublayers output 512-dimensional vectors, allowing residual addition. The decoder inputs are shifted right by one token; together with causal masking, this prevents predictions from accessing the target token being predicted or later tokens. Training processes target positions in parallel, while generation proceeds sequentially.

### 3.2 Attention

Attention compares a query with keys to obtain weights, then returns a weighted sum of the corresponding values.

#### 3.2.1 Scaled dot-product attention

<p align="center">
  <img src="./images/figure2.1.png" alt="Scaled dot-product attention" width="250">
</p>

*Figure 2 (left). Scaled dot-product attention.*

Queries and keys have dimension $`d_k`$; values have dimension $`d_v`$. With queries, keys, and values packed into matrices:

```math
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
```

Softmax operates across each row of the score matrix. In self-attention, the matrix has shape $`n\times n`$, connecting every query position to every key position.

Assuming independent query and key components with zero mean and unit variance, their dot product has variance $`d_k`$. Dividing by $`\sqrt{d_k}`$ prevents large scores from pushing softmax into regions with small gradients. Dot-product attention also benefits from efficient matrix multiplication.

#### 3.2.2 Multi-head attention

<p align="center">
  <img src="./images/figure2.2.png" alt="Multi-head attention" width="350">
</p>

*Figure 2 (right). Multi-head attention.*

Each head applies separate learned projections before computing attention. The outputs are concatenated and projected:

```math
\mathrm{head}_i=\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V).
```

```math
\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O.
```

Different heads can attend to different positions and representation subspaces. The base model uses eight heads with $`d_k=d_v=64`$. Their outputs concatenate to 512 dimensions. Smaller dimensions per head keep the total computational cost similar to one head at full dimensionality.

#### 3.2.3 Applications of attention

- **Encoder-decoder attention:** Queries come from the decoder; keys and values come from the final encoder output.
- **Encoder self-attention:** Queries, keys, and values come from the same encoder representation; every source position can attend to every source position.
- **Decoder self-attention:** Each position attends only to itself and earlier positions in the shifted target input.

Causal masking sets forbidden scores to $`-\infty`$ before softmax, assigning future positions zero attention weight.

### 3.3 Position-wise feed-forward networks

Each layer applies two linear transformations with a ReLU between them:

```math
\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2.
```

The network expands from 512 to 2048 dimensions and returns to 512. Parameters are shared across positions within a layer but differ between layers. Attention mixes information across tokens; the feed-forward network transforms each token independently.

### 3.4 Embeddings and softmax

Learned embeddings map source and target tokens to 512-dimensional vectors. A linear transformation and softmax convert decoder outputs into next-token probabilities. The paper shares weights between both embedding layers and the output projection, and scales embedding vectors by $`\sqrt{d_{\mathrm{model}}}`$.

### 3.5 Positional encoding

Because self-attention alone does not encode order, positional vectors are added to embeddings at the bottoms of both stacks:

```math
PE_{(p,2j)}=\sin\left(\frac{p}{10000^{2j/d_{\mathrm{model}}}}\right).
```

```math
PE_{(p,2j+1)}=\cos\left(\frac{p}{10000^{2j/d_{\mathrm{model}}}}\right).
```

Here $`p`$ denotes position and $`j`$ indexes sine-cosine pairs. Different frequencies encode position at different scales. For fixed offset $`k`$, the encoding at $`p+k`$ is a linear transformation of the encoding at $`p`$, motivating their use for relative positions. Learned positional embeddings performed similarly; sinusoids were chosen for possible extrapolation to longer sequences.

### Training 

The model was trained on the WMT 2014 English-German dataset and the WMT 2014 English-French dataset, containing 4.5 million and 36 million sentences each. The Adam optimizer, residual dropout and label smoothing was used. The model was trained on eight NVIDIA P100 GPUs. 

## Results

### Translation quality and efficiency

The Transformer achieves state-of-the-art BLEU scores on both English-to-German and English-to-French translation tasks as the fraction of the training cost. 

<p align="center">
  <img src="images/table2.jpeg" alt="Translation BLEU scores and training costs" width="600">
</p>

<p align="center"><em>Table 2. BLEU scores and training costs of English-to-German and English-to-French translation models.</em></p>

### English Constituency Parsing

In order to evaluate the Transformer for other NLP tasks, its performance on English constituency parsing was experimented. The Wall Street Journal (WSJ) portion of the Penn Treebank and BerkeleyParser datasets were used. Despite the lack of task-specific tuning, the Transformer outperformed all previous models except one. 

<p align="center">
  <img src="images/table4.jpeg" alt="English constituency parsing results" width="600">
</p>

<p align="center"><em>Table 4. English constituency parsing results. </em></p>

