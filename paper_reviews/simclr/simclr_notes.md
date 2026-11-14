# [Paper Review] A Simple Framework for Contrastive Learning of Visual Representations

> **Authors:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton\
> **Institutions:** Google Research, Brain Team\
> **Publication:** ICML 2020\
> **Link:** [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

Contrastive learning produces transferable visual features without semantic labels.

## 1 Introduction

The paper isolates augmentation, projection, and optimization choices.

## 2 Method

### 2.1 The Contrastive Learning Framework

Two augmented views share an encoder $f$ and projection head $g$. Optimize projected features; retain encoder features downstream.

```math
h_i=f(\widetilde x_i),\quad z_i=g(h_i),\qquad
\ell_{i,j}=-\log
\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
{\sum_{k\ne i}^{2N}\exp(\operatorname{sim}(z_i,z_k)/\tau)}.
```

The denominator includes the positive; the batch supplies negatives.

### 2.2 Training with Large Batch Size

LARS and global batch normalization support large-batch training.

### 2.3 Evaluation Protocol

Train a linear classifier over a frozen encoder.

## 3 Data Augmentation for Contrastive Representation Learning

### 3.1 Composition of Data Augmentation Operations

Cropping plus color distortion reduces shortcut learning.

### 3.2 Stronger Augmentation than Supervised Learning

Effective augmentation depends on the learning objective.

## 4 Architectures for Encoder and Head

### 4.1 Bigger Models

Depth and width improve representations.

### 4.2 Nonlinear Projection Head

Features before the head transfer better than projected features.

## 5 Loss Functions and Batch Size

### 5.1 Normalization and Temperature

Both affect contrastive learning quality.

### 5.2 Larger Batches and Longer Training

Both help; longer training mitigates smaller batches.

## 6 Comparison with State-of-the-art

Evaluate linear probing, limited-label fine-tuning, and transfer.

## 7 Related Work

Contrast with memory-bank and pretext-task methods.

## 8 Conclusion

Simple components jointly enable strong self-supervised learning.
