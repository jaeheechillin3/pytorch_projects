# [Paper Review] Neural Machine Translation by Jointly Learning to Align and Translate

> **Authors:** Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio\
> **Institutions:** Jacobs University Bremen, Université de Montréal\
> **Publication:** ICLR 2015\
> **Link:** https://arxiv.org/abs/1409.0473

## Abstract & Introduction

Previous machine translation systems consisted of many smaller subcomponents. A new approach called _neural machine translation_ uses a single neural network to read a sentence and output a translation. Most neural machine translation models are encoder-decoders where a sentence is transformed into a _fixed-length_ encoding vector, from which the translation is outputted. The problem with this approach is that it is difficult to compress all the information into a fixed-length vector for very long sentences. 

To address this issue, the paper proposes an "attention" mechanism that for each target word, identifies the most relevant words in the original sentence. Each new target word is produced based on its "context vectors" as well as the previously generated target words.

## Methods

### Encoder representations

The encoder retains one representation per source word instead of returning only its final state. Each annotation contains the word's surrounding context, so it is more informative than an isolated word embedding. For example, the representation of a word with several meanings can depend on the rest of its sentence.

A bidirectional recurrent encoder concatenates forward and backward states into an annotation $h_j$ for each source position. The decoder's previous state $s_{i-1}$ scores each annotation through a learned alignment network:

$$
e_{ij}=v_a^\top\tanh(W_as_{i-1}+U_ah_j),\qquad
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_k\exp(e_{ik})}.
$$

### Alignment and context

Here, $i$ indexes target words and $j$ indexes source words. The alignment score asks how useful source position $j$ is for producing target position $i$. Parameters $W_a$, $U_a$, and $v_a$ are shared across positions; the model does not learn a separate network for every word pair.

The weights sum to one across source positions, so the decoder can focus on several relevant words rather than selecting a single hard alignment.

The context vector is a weighted sum:

$$
c_i=\sum_j\alpha_{ij}h_j,\qquad
s_i=f(s_{i-1},y_{i-1},c_i).
$$

The dimensionality of $c_i$ is fixed, but its contents change at every decoding step. Attention therefore does not eliminate fixed-size vectors altogether: it eliminates the requirement that one unchanged vector carry the entire sentence throughout decoding.

### Prediction and learning

The decoder predicts $y_i$ using its state, context, and previous word. Soft attention is differentiable, so alignment and translation are learned together by maximizing translation log-likelihood without annotated alignments.

The conditional distribution factorizes over target positions:

$
p(y\mid x)=\prod_{i=1}^{T_y}p(y_i\mid y_{<i},x),\qquad
\mathcal J=-\sum_i\log p(y_i\mid y_{<i},x).
$

Training minimizes this negative log-likelihood. A prediction error can update both the decoder and the alignment network, teaching the model where to retrieve information. At inference, previous target words must come from the model's own predictions; the paper uses beam search to retain several candidate translations.

## Results

### Translation quality

On English-to-French translation, **RNNsearch-50 scores 26.75 BLEU**, compared with **17.82** for RNNencdec-50 (Table 1). Longer training raises RNNsearch to 28.45; Moses still leads on the full test set at 33.30.

The “50” suffix refers to the maximum training sentence length, not the number of layers. The comparison tests attention against a fixed-vector encoder–decoder trained under the same data and training procedures. On the subset without unknown words, the longer-trained RNNsearch reaches 36.15 BLEU versus Moses's 35.63. This narrower result should not be confused with winning on the full test set.

### Sentence length and alignment

Attention substantially reduces degradation on long sentences (Figure 2). Alignment plots show plausible word correspondences and reordering (Figure 3).

A bright entry in an alignment plot means a large weight for a particular source–target pair. Off-diagonal patterns can reflect differences in word order between languages. These plots help inspect the learned retrieval mechanism, although attention weights are not guaranteed linguistic annotations.

The contribution is selective access to source representations; recurrent computation and unknown-word limitations remain.
