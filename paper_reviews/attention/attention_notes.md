# [Paper Review] Neural Machine Translation by Jointly Learning to Align and Translate

> **Authors:** Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio\
> **Institutions:** Jacobs University Bremen, Université de Montréal\
> **Publication:** ICLR 2015\
> **Link:** https://arxiv.org/abs/1409.0473

## Abstract & 1 Introduction

Previous machine translation systems consisted of many smaller subcomponents. A new approach called _neural machine translation_ uses a single neural network to read a sentence and output a translation. Most neural machine translation models are encoder-decoders where a sentence is transformed into a _fixed-length_ encoding vector, from which the translation is outputted. The problem with this approach is that it is difficult to compress all the information into a fixed-length vector for very long sentences. 

To address this issue, the paper proposes an "attention" mechanism that for each target word, identifies the most relevant words in the original sentence. Each new target word is produced based on its "context vectors" as well as the previously generated target words.

## 2 Background: Neural Machine Translation

### 2.1 RNN Encoder–Decoder

A conventional encoder reads the source sentence into one fixed-length context vector. A recurrent decoder then generates target words conditioned on that vector and earlier target words:

```math
p(y\mid x)=\prod_{i=1}^{T_y}p(y_i\mid y_1,\ldots,y_{i-1},x).
```

Training maximizes the probability of reference translations, equivalently minimizing negative log-likelihood. The bottleneck is requiring one unchanged context vector to preserve all source information throughout decoding.

## 3 Learning to Align and Translate

> **ADD FIGURE 1 — model architecture.** Save the original paper's Figure 1 as **images/figure1.png**. It shows the bidirectional encoder, attention weights, and context used to generate a target word.

### 3.1 Decoder: General Description

The decoder uses a different context vector for every target position:

```math
p(y_i\mid y_1,\ldots,y_{i-1},x)=g(y_{i-1},s_i,c_i).
```

```math
s_i=f(s_{i-1},y_{i-1},c_i).
```

Here $`s_i`$ is the decoder state, and $`c_i`$ retrieves information from the source annotations:

```math
c_i=\sum_{j=1}^{T_x}\alpha_{ij}h_j.
```

The alignment network scores each annotation using the previous decoder state. Softmax normalizes these scores over source positions:

```math
e_{ij}=a(s_{i-1},h_j),
\qquad
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})}.
```

Here $`i`$ indexes target words and $`j`$ indexes source words. The alignment score asks how useful source position $`j`$ is for producing target word $`i`$. Appendix A specifies the additive scoring function:

```math
a(s_{i-1},h_j)=v_a^\top\tanh(W_as_{i-1}+U_ah_j).
```

Parameters are shared across positions. The weights sum to one, allowing attention to several relevant words rather than one hard alignment. Because this weighted sum is differentiable, alignment and translation are learned jointly without annotated alignments.

The dimension of $`c_i`$ remains fixed, but its contents change at each step. Attention removes the requirement that one unchanged vector carry the entire sentence.

### 3.2 Encoder: Bidirectional RNN for Annotating Sequences

The encoder runs recurrent networks in both directions and concatenates their states at each source position:

```math
h_j=\begin{bmatrix}\overrightarrow{h}_j\\\overleftarrow{h}_j\end{bmatrix}.
```

Each annotation contains information from preceding and following words, with a strong focus on the surrounding source region. The decoder retains access to the full sequence of annotations rather than only a final encoder state.

## 4 Experiment Settings

### 4.1 Dataset

Experiments use WMT 2014 English-to-French parallel data. Each language has a 30,000-word vocabulary; words outside it become [UNK] tokens. Evaluation uses newstest2014.

### 4.2 Models

The paper compares a conventional fixed-vector **RNNencdec** with the attention-based **RNNsearch**, using the same dataset and training procedures. The suffixes “30” and “50” indicate maximum training sentence lengths, not layer counts.

RNNsearch uses gated recurrent units, a bidirectional encoder, and a recurrent decoder. Training uses Adadelta; inference uses beam search to retain several candidate translations.

## 5 Results

### 5.1 Quantitative Results

> **ADD TABLE 1 — BLEU comparison.** Save the original paper's Table 1 as **images/table1.png**. Include both the **All** and **No UNK** columns and the note identifying the longer-trained RNNsearch-50 model.

**RNNsearch-50 scores 26.75 BLEU**, compared with **17.82** for RNNencdec-50. Longer training raises RNNsearch to 28.45, while Moses still leads on the full test set at 33.30.

On the restricted evaluation without unknown words, the longer-trained RNNsearch reaches 36.15 BLEU versus Moses's 35.63. This does not mean it wins on the full test set.

> **ADD FIGURE 2 — BLEU versus sentence length.** Save the original paper's Figure 2 as **images/figure2.png**. Keep the axes and legend visible; this is the main evidence for improved handling of long sentences.

Attention substantially reduces the degradation seen in the fixed-vector model as sentences grow longer. RNNsearch-30 even outperforms RNNencdec-50 overall.

### 5.2 Qualitative Analysis

#### 5.2.1 Alignment

> **ADD FIGURE 3 — learned alignment heatmaps.** Save the full four-panel Figure 3 as **images/figure3.png**, including source and target words. Panels **(a)** and **(d)** are especially useful for discussing reordering and attention to multiple words.

A bright cell represents a large attention weight for a source–target pair. Diagonal patterns indicate similar word order; off-diagonal patterns can reflect reordering. Soft alignment also lets a target word draw on multiple source words. These plots show plausible correspondences, although attention weights are not guaranteed linguistic annotations.

#### 5.2.2 Long Sentences

The paper examines translations in which the fixed-vector model loses information or deviates from the source meaning, while RNNsearch preserves more details. These examples complement the sentence-length results in Figure 2.

## 6 Related Work

### 6.1 Learning to Align

Earlier attention for handwriting synthesis constrained alignment to move forward. Translation requires more flexible reordering, so this model scores every source position at each decoding step.

### 6.2 Neural Networks for Machine Translation

Earlier systems often used neural networks to score phrases or rerank translations. This work generates translations directly through a jointly trained neural model.

## 7 Conclusion

Attention gives the decoder selective access to source representations and improves translation, especially for long sentences. Recurrent computation remains sequential, and unknown or rare words remain an important limitation.
