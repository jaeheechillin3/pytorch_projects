# [Paper Review] A Simple Framework for Contrastive Learning of Visual Representations

> **Authors:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton\
> **Institutions:** Google Research, Brain Team\
> **Publication:** ICML 2020\
> **Link:** [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)


## Abstract & Introduction

Self-supervised learning aims to learn useful image representations without class labels. Contrastive methods do this by bringing related views together and separating unrelated examples. Earlier methods often required a memory bank or specialized model components.

This paper introduces SimCLR, a framework built from data augmentation, an image encoder, a projection head, and a contrastive loss. Its experiments examine which choices make this simple setup effective. The composition of augmentations, a nonlinear projection head, and a large training batch all contribute substantially to representation quality.

## Methods

### The contrastive learning framework

Two independently sampled transformations are applied to each image. These produce a positive pair: two views of the same original image. Each view passes through the same ResNet encoder $f$, followed by a small projection network $g$:

```math
h_i=f(\widetilde x_i),\qquad z_i=g(h_i).
```

The encoder representation $h_i$ is the output after global average pooling. The projection head is a two-layer MLP with a ReLU activation. The contrastive objective acts on $z_i$; downstream tasks use $h_i$ after discarding the head.

<p align="center">
  <img src="https://1.bp.blogspot.com/--vH4PKpE9Yo/Xo4a2BYervI/AAAAAAAAFpM/vaFDwPXOyAokAC8Xh852DzOgEs22NhbXwCLcBGAsYHQ/s1600/image4.gif" alt="Two augmented views pass through a shared encoder and projection head" width="600">
</p>

<p align="center"><em>SimCLR framework. Illustration from the authors' implementation.</em></p>

For a minibatch of $N$ images, augmentation creates $2N$ views. Each view has one positive counterpart and $2N-2$ negatives from the other images. The normalized temperature-scaled cross entropy loss, or NT-Xent, is

```math
\ell_{i,j}=-\log
\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
{\sum_{k=1}^{2N}\mathbf 1_{[k\ne i]}
\exp(\operatorname{sim}(z_i,z_k)/\tau)}.
```

Here $\operatorname{sim}$ is cosine similarity and $\tau$ is the temperature. The denominator includes the positive view and excludes the anchor itself. Both directions of every positive pair contribute to the batch loss:

```math
\mathcal L=\frac{1}{2N}\sum_{k=1}^{N}
\left[\ell_{2k-1,2k}+\ell_{2k,2k-1}\right].
```

### Data augmentation

The augmentation pipeline combines random cropping and resizing, color distortion, and Gaussian blur. Cropping creates views that can contain different parts of an object or different amounts of context. Color distortion makes it harder to identify a positive pair by its color distribution alone.

The combination matters more than any single transformation. Without strong color changes, the network can exploit similarities between the color histograms of crops. This solves the training task without requiring the representations needed for recognition. Stronger augmentation works particularly well for contrastive learning, even when the same transformations would be less useful in supervised training.

### Encoder and projection head

The paper studies deeper and wider ResNets. Increasing model capacity improves the learned representations. A nonlinear projection head also improves the representation before the head, even though the head is discarded afterward.

One explanation is that the contrastive loss encourages invariance to the chosen transformations. Applying all of that pressure directly to the downstream representation can remove useful information. The head provides a separate space in which to enforce agreement.

### Training and evaluation

Training uses large batches, with up to 8,192 original images, and the LARS optimizer. Global batch normalization shares statistics across devices, preventing a network from exploiting local batch statistics to identify matching views. Larger batches provide more negatives per update. Longer training also helps and reduces some of the gap between smaller and larger batches.

The main evaluation freezes the encoder and trains a linear classifier on its representations. Additional experiments fine-tune the model with limited labels and evaluate transfer to other datasets.

## Results

### ImageNet classification

SimCLR achieves **76.5% ImageNet top-1 accuracy** with a linear classifier on a four-times-wider ResNet-50. This headline result uses a wider ResNet-50, rather than the standard-width model. It matches the reported supervised standard ResNet-50 baseline despite learning the encoder without ImageNet labels.

### Limited labels and transfer

Fine-tuning with only 1% of the ImageNet labels reaches **85.8% top-5 accuracy**. Transfer experiments on twelve natural-image classification datasets find that SimCLR matches or exceeds the supervised baseline on ten.

### Ablation studies

Removing feature normalization or choosing an unsuitable temperature reduces performance. The nonlinear projection head outperforms a linear head or no head. Augmentation composition and sufficient training time also have large effects. These experiments explain why a compact framework can perform well: the training task and optimization choices determine the quality of the representation.
