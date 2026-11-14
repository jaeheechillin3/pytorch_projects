# [Paper Review] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

> **Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova\
> **Institutions:** Google AI Language\
> **Publication:** NAACL 2019\
> **Link:** [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

Bidirectional Transformer pretraining supplies reusable representations for multiple language-understanding tasks.

## 1 Introduction

Masked prediction enables both left and right context during pretraining.

## 2 Related Work

### 2.1 Unsupervised Feature-based Approaches

ELMo transfers contextual features into task-specific models.

### 2.2 Unsupervised Fine-tuning Approaches

GPT transfers pretrained parameters through fine-tuning.

### 2.3 Transfer Learning from Supervised Data

Supervised pretraining is another route to transferable representations.

## 3 BERT

A bidirectional encoder consumes summed token, segment, and position embeddings.

### 3.1 Pre-training BERT

**Masked LM.** Select 15% of tokens; replace selected tokens with [MASK], random tokens, or unchanged tokens in an 80/10/10 ratio. Predict originals at selected positions.

```math
\mathcal L_{\mathrm{MLM}}=-\sum_{i\in M}\log p_\theta(x_i\mid\widetilde{x}).
```

**Next sentence prediction.** Classify genuine versus randomly paired continuations, sampled equally.

### 3.2 Fine-tuning BERT

Attach task-specific outputs and update all parameters.

## 4 Experiments

### 4.1 GLUE

Evaluate sentence classification and sentence-pair understanding.

### 4.2 SQuAD v1.1

Predict answer-span endpoints.

### 4.3 SQuAD v2.0

Also recognize unanswerable questions.

### 4.4 SWAG

Select plausible continuations.

## 5 Ablation Studies

### 5.1 Effect of Pre-training Tasks

Compare bidirectionality and NSP ablations.

### 5.2 Effect of Model Size

Larger models improve downstream results.

### 5.3 Feature-based Approach with BERT

Test frozen features on named-entity recognition.

## 6 Conclusion

Pretraining reduces task-specific architectural engineering.

## Appendix notes

Masking alternatives, training schedules, and fine-tuning details complement the main comparisons.
