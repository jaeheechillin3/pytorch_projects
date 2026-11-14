# [Paper Review] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

> **Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova\
> **Institutions:** Google AI Language\
> **Publication:** NAACL 2019\
> **Link:** [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

## Abstract & Introduction 

This paper introduces the Bidirectional Encoder Representations from Transformers, or BERT. The model incorporates both left and right context to produce rich representations. After being pretrained on unlabeled text, it can be fine-tuned for a large variety of natural language processing tasks. This approach of pretraining followed by fine-tuning can replace and outperform many task-specific architectures; BERT achieves state-of-the-art performance on eleven different NLP tasks. 

## Methods 

A bidirectional encoder consumes summed token, segment, and position embeddings. [CLS] supplies a sequence-level output, while [SEP] separates packed segments. BERT-Base uses 12 layers, hidden width 768, and 12 attention heads. BERT-Large uses 24 layers, width 1024, and 16 heads. WordPiece tokenization represents inputs as subword units. Segment embeddings identify which of two packed text spans a token belongs to; position embeddings supply sequence order.

<p align="center">
  <!-- TODO: Add ./images/figure2.png, then uncomment the image below. -->
  <!-- <img src="./images/figure2.png" alt="Token, segment, and position embeddings are summed." width="600"> -->
</p>

<p align="center"><em>Figure 2. Token, segment, and position embeddings are summed. [Image to add]</em></p>

### 3.1 Pre-training BERT

**Masked LM.** Select 15% of tokens; replace selected tokens with [MASK], random tokens, or unchanged tokens in an 80/10/10 ratio. Predict originals at selected positions. Supervise every selected position, including unchanged tokens; 15% denotes selection, not [MASK] replacement. Keeping some selected tokens unchanged and replacing others randomly reduces reliance on a special token absent during downstream use. Unlike ordinary next-token prediction, the objective only scores the selected positions.

```math
\mathcal L_{\mathrm{MLM}}=-\sum_{i\in M}\log p_\theta(x_i\mid\widetilde{x}).
```

Here $M$ is the selected position set and $\widetilde{x}$ is the corrupted sequence. Bidirectional attention is allowed over that sequence; the original answers remain training targets rather than inputs at replaced positions.

**Next sentence prediction.** Classify genuine versus randomly paired continuations, sampled equally. BooksCorpus and Wikipedia supply pretraining examples. The [CLS] output predicts whether the second span actually follows the first. A “sentence” here may be a contiguous text span rather than a linguistically complete sentence. This auxiliary task trains relationships between paired inputs.

### 3.2 Fine-tuning BERT

Attach task-specific outputs and update all parameters. Use [CLS] for classification and token outputs for tagging. Each downstream task gets its own fine-tuned model initialized from the same pretrained weights. Packing a question and passage together lets self-attention model their interactions directly, without a separately designed cross-attention system.

<p align="center">
  <!-- TODO: Add ./images/figure1.png, then uncomment the image below. -->
  <!-- <img src="./images/figure1.png" alt="Shared pretrained encoder adapted to different downstream tasks." width="600"> -->
</p>

<p align="center"><em>Figure 1. Shared pretrained encoder adapted to different downstream tasks. [Image to add]</em></p>

## 4 Experiments

### 4.1 GLUE

Evaluate sentence classification and sentence-pair understanding. BERT-Large's official GLUE score is 80.5; Table 1 excludes WNLI from its average. The benchmark mixes different tasks and metrics, so its aggregate should not be interpreted as one ordinary classification accuracy. The broad gains support the usefulness of shared pretraining beyond a single application.

<p align="center">
  <!-- TODO: Add ./images/table1.png, then uncomment the image below. -->
  <!-- <img src="./images/table1.png" alt="GLUE test results for BERT and baseline systems." width="650"> -->
</p>

<p align="center"><em>Table 1. GLUE test results for BERT and baseline systems. [Image to add]</em></p>

### 4.2 SQuAD v1.1

Predict answer-span endpoints using separate start/end scores over passage tokens. A softmax over token scores defines each endpoint distribution, and training increases the probabilities of the annotated endpoints. At inference, the selected span must have its end at or after its start.

### 4.3 SQuAD v2.0

Also recognize unanswerable questions by comparing the best span against a [CLS]-based no-answer score. Unanswerable training examples assign both endpoints to [CLS]. A development-set threshold controls whether to return a passage span or no answer.

### 4.4 SWAG

Score each context–continuation pair and select among four choices. Each candidate becomes a separate packed input, and a shared scorer operates on its [CLS] representation.

## 5 Ablation Studies

### 5.1 Effect of Pre-training Tasks

Compare bidirectionality and NSP ablations. Removing NSP reduces several scores; replacing masked pretraining with left-to-right modeling particularly hurts token-level question answering. The ablations keep the training data and fine-tuning scheme controlled, helping separate objective choices from dataset differences. Their NSP findings describe this experimental setup rather than establishing a universal requirement for every later model.

<p align="center">
  <!-- TODO: Add ./images/table5.png, then uncomment the image below. -->
  <!-- <img src="./images/table5.png" alt="Pretraining-objective ablations with the BERT-Base architecture." width="650"> -->
</p>

<p align="center"><em>Table 5. Pretraining-objective ablations with the BERT-Base architecture. [Image to add]</em></p>

### 5.2 Effect of Model Size

Larger models improve downstream results, including tasks with few labeled examples. This suggests that increasing pretrained capacity can help even when the downstream dataset alone would be too small to train such a model effectively.

### 5.3 Feature-based Approach with BERT

Test frozen features on named-entity recognition as an alternative to full fine-tuning. Frozen contextual representations remain useful, showing that the pretrained model can also supply features when updating all its parameters is undesirable.
