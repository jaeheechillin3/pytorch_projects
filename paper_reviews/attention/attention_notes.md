# [Paper Review] Neural Machine Translation by Jointly Learning to Align and Translate

> **Authors:** Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio\
> **Institutions:** Jacobs University Bremen, Université de Montréal\
> **Publication:** ICLR 2015\
> **Link:** https://arxiv.org/abs/1409.0473

## Abstract & Introduction

Early neural encoder–decoder models compress an entire source sentence into one fixed-length vector, making long sentences difficult to translate. This paper introduces attention: the decoder retrieves different source information for each target word.

## Methods

A bidirectional recurrent encoder concatenates forward and backward states into an annotation $h_j$ for each source position. The decoder's previous state $s_{i-1}$ scores each annotation through a learned alignment network:

$$
e_{ij}=v_a^\top\tanh(W_as_{i-1}+U_ah_j),\qquad
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_k\exp(e_{ik})}.
$$

The context vector is a weighted sum:

$$
c_i=\sum_j\alpha_{ij}h_j,\qquad
s_i=f(s_{i-1},y_{i-1},c_i).
$$

The decoder predicts $y_i$ using its state, context, and previous word. Soft attention is differentiable, so alignment and translation are learned together by maximizing translation log-likelihood without annotated alignments.

## Results

On English-to-French translation, **RNNsearch-50 scores 26.75 BLEU**, compared with **17.82** for RNNencdec-50 (Table 1). Longer training raises RNNsearch to 28.45; Moses still leads on the full test set at 33.30.

Attention substantially reduces degradation on long sentences (Figure 2). Alignment plots show plausible word correspondences and reordering (Figure 3).

The contribution is selective access to source representations; recurrent computation and unknown-word limitations remain.
