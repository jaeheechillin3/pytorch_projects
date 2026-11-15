# [Paper Review] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

> **Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova\
> **Institutions:** Google AI Language\
> **Publication:** NAACL 2019\
> **Link:** [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)


## Abstract & Introduction

This paper introduces Bidirectional Encoder Representations from Transformers, or BERT. The model incorporates both left and right context to produce representations that can be adapted to many natural language processing tasks. After pretraining on unlabeled text, the same model architecture is fine-tuned with a small task-specific output layer.

Earlier fine-tuning approaches such as GPT used left-to-right language modeling. Each token could therefore attend only to earlier tokens, limiting the context available for tasks such as question answering. ELMo combined separately trained forward and backward language models. BERT instead allows both directions to interact at every layer.

BERT achieves state-of-the-art results on eleven NLP tasks. Its contribution is a pretraining procedure that makes a deeply bidirectional Transformer useful across sentence-level and token-level applications.

## Methods

### Model architecture and input representations

BERT uses the encoder from the original Transformer. BERT-Base has 12 layers, hidden width 768, and 12 attention heads, with 110 million parameters. BERT-Large has 24 layers, hidden width 1,024, and 16 heads, with 340 million parameters.

The input can contain one text span or a pair. A [CLS] token is added at the beginning, and [SEP] tokens mark boundaries. The final [CLS] representation is used for sequence classification. A question and passage can be packed together so that their tokens interact directly through self-attention.

Each input representation sums three learned embeddings:

```math
E_i=E_{\mathrm{token},i}+E_{\mathrm{segment},i}+E_{\mathrm{position},i}.
```

WordPiece tokenization uses a 30,000-token vocabulary. Segment embeddings distinguish the two packed spans, while position embeddings encode sequence order. In the paper, a “sentence” can be a contiguous span of text rather than a complete grammatical sentence.

### Masked language modeling

A conventional bidirectional language model could see the token it is supposed to predict. BERT avoids this problem by selecting 15% of input positions for prediction. Of the selected tokens, 80% are replaced with [MASK], 10% with a random token, and 10% remain unchanged.

The objective predicts the original tokens at all selected positions:

```math
\mathcal L_{\mathrm{MLM}}
=-\sum_{i\in M}\log p_\theta(x_i\mid\widetilde x).
```

Here $M$ is the selected position set and $\widetilde x$ is the corrupted input. Only selected positions contribute to this loss. Random replacements and unchanged tokens reduce the mismatch caused by [MASK] appearing during pretraining but not ordinary fine-tuning.

Because prediction uses the corrupted sequence, the encoder can attend bidirectionally without having access to the original answers at replaced positions.

### Next sentence prediction

The second objective classifies whether a second span follows the first in the source document. Half of the training pairs are genuine continuations; the other half use a randomly sampled span. A classifier over [CLS] predicts _IsNext_ or _NotNext_.

This next sentence prediction (NSP) objective trains a representation of the relationship between paired inputs. The total pretraining loss combines MLM and NSP.

### Pretraining and fine-tuning

Pretraining uses BooksCorpus, containing 800 million words, and English Wikipedia, containing 2.5 billion words. Document boundaries are retained so that training examples can contain long contiguous passages.

The models train for one million steps with batches of 256 sequences. Most steps use sequences of length 128, followed by length 512 for the final 10%. This reduces the cost of pretraining while still training the longer position embeddings.

Fine-tuning initializes the encoder from these weights and updates all parameters. Sequence classification attaches an output layer to [CLS]. Token classification attaches outputs to individual token states. Each downstream task produces its own fine-tuned model.

## Results

### GLUE

BERT improves results across sentence classification and sentence-pair tasks. BERT-Large achieves an official GLUE score of **80.5**. The paper's Table 1 reports individual tasks and an average excluding WNLI, so that table average differs from the official benchmark score.

### SQuAD v1.1 and v2.0

For extractive question answering, the input combines a question with its passage. Learned vectors score each passage token as a possible start or end of the answer:

```math
P_{\mathrm{start}}(i)=
\frac{\exp(S^\top T_i)}{\sum_j\exp(S^\top T_j)},\qquad
P_{\mathrm{end}}(i)=
\frac{\exp(E^\top T_i)}{\sum_j\exp(E^\top T_j)}.
```

The predicted answer is a valid span with a high combined endpoint score. The SQuAD v1.1 ensemble achieves **93.2 test F1**.

SQuAD v2.0 also contains unanswerable questions. Training uses [CLS] as the start and end for these examples. At inference, its score is compared with the best answer span using a tuned threshold. BERT-Large reaches **83.1 test F1**.

### SWAG

Each of four candidate continuations is paired with the context and scored from [CLS]. Selecting the highest score gives **86.3% test accuracy** for BERT-Large.

### Ablation studies

Removing NSP reduces performance on several tasks. In the BERT-Base ablation, QNLI accuracy falls from 88.4 to 84.9. Replacing masked pretraining with left-to-right modeling hurts more strongly on token-level tasks: SQuAD F1 falls from 88.5 to 77.8.

Larger pretrained models also improve downstream results, including on MRPC, which has only about 3,600 training examples. A feature-based experiment on named-entity recognition finds that frozen BERT features remain useful, although full fine-tuning performs better.

The paper's [Figures 1–2 and Tables 1–6](https://arxiv.org/pdf/1810.04805) present the input format, task adaptations, and benchmark comparisons.
