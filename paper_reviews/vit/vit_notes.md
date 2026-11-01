# [Paper Review] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

> **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby\
> **Institutions:** Google Research, Brain Team\
> **Publication:** ICLR 2021\
> **Link:** https://arxiv.org/abs/2010.11929

## Abstract & 1 Introduction

In natural language processing (NLP), Transformers had become a dominant architecture by the time of this paper. Through pretraining on a large text corpus then fine-tuning, Transformers have shown strong performance on many NLP tasks. However, convolutional neural networks (CNNs) remained the dominant architecture at the time in computer vision. There have been multiple attempts to introduce the self-attention mechanism to various parts of CNNs, often while retaining a convolutional backbone. 

This paper takes a different approach; instead of relying on CNNs, Transformers are applied directly to supervised image classification with as few modifications from the original architecture as possible. Images are split into patches, each of which is treated the same way as tokens in NLP tasks. 

On smaller pretraining datasets, Vision Transformers (ViTs) lag behind in accuracy compared to CNNs of comparable sizes. This is due to ViTs having fewer built-in inductive biases like translation equivariance and locality, both of which are inherent to CNNs. However, ViTs perform at the state of the art level on large datasets.

## 2 Related Work

Earlier approaches combined CNNs with attention or restricted attention to local regions to reduce computation. ViT instead uses image patches as tokens, allowing a standard Transformer encoder to process images efficiently.

## 3 Method

### 3.1 Vision Transformer (ViT)

<p align="center">
  <img src="./images/figure1.jpg" alt="Vision Transformer patch embedding and encoder architecture" width="650">
</p>

<p align="center"><em>Figure 1. Patch embeddings and a class token enter a Transformer encoder.</em></p>

**Patch and position embeddings.** An image of size $`H\times W`$ with $`C`$ channels is split into $`N=HW/P^2`$ non-overlapping patches. Each flattened patch contains $`P^2C`$ values and is projected into a $`D`$-dimensional embedding. A learnable class token is prepended, and learned 1D position embeddings are added:

```math
z_0=[x_{\mathrm{class}};x_p^1E;\ldots;x_p^NE]+E_{\mathrm{pos}}.
```

```math
E\in\mathbb{R}^{(P^2C)\times D},
\qquad E_{\mathrm{pos}}\in\mathbb{R}^{(N+1)\times D}.
```

For example, a $`224\times224`$ RGB image with $`16\times16`$ patches yields 196 patches of 768 values each, or 197 tokens including the class token. Patch size and embedding dimension are separate choices. Smaller patches produce longer sequences and increase attention cost.

**Transformer encoder and classification.** Each of the $`L`$ encoder layers contains multi-head self-attention (MSA) and a two-layer MLP with GELU. Layer normalization precedes each sublayer, and residual connections follow:

```math
z'_\ell=\mathrm{MSA}(\mathrm{LN}(z_{\ell-1}))+z_{\ell-1}.
```

```math
z_\ell=\mathrm{MLP}(\mathrm{LN}(z'_\ell))+z'_\ell.
```

The final class-token state becomes the image representation after normalization:

```math
y=\mathrm{LN}(z_L^0).
```

The input class token is shared across images, but its output becomes image-specific through attention. A classification head uses this representation: an MLP with one hidden layer during pretraining and a linear layer during fine-tuning. ViT uses an encoder for classification, so no causal mask or autoregressive decoder is needed.

**Inductive bias.** CNNs build locality, neighborhood structure, and translation equivariance into their layers. ViT uses global attention and learns spatial relationships through position embeddings. Its weaker image-specific assumptions make learning harder when pretraining data are scarce.

**Hybrid architecture.** The paper also considers tokens extracted from CNN feature maps rather than raw image patches, followed by the same embedding and Transformer processing.

### 3.2 Fine-tuning and Higher Resolution

Models are pretrained on large labeled datasets and adapted to downstream tasks by replacing the prediction head. Higher-resolution fine-tuning keeps patch size fixed, producing more tokens. The spatial position embeddings are interpolated in two dimensions to match the new grid; the class-token position embedding is kept separate.

## 4 Experiments

### 4.1 Setup

Pretraining uses ImageNet, ImageNet-21k, or JFT-300M. Evaluation includes ImageNet, CIFAR-10/100, Pets, Flowers, and VTAB. The paper compares ViT-Base, Large, and Huge with ResNet (BiT) and hybrid models. In names such as ViT-L/16, the letter indicates model size and the number indicates patch width.

Pretraining uses Adam; fine-tuning uses SGD with momentum. Results include both full fine-tuning and linear evaluation of frozen representations.

### 4.2 Comparison to State of the Art

JFT-pretrained **ViT-H/14 reaches 88.55% ImageNet and 94.55% CIFAR-100 accuracy** after fine-tuning.

<p align="center">
  <img src="./images/table2.jpeg" alt="Fine-tuning accuracy and pretraining cost across image classification benchmarks" width="650">
</p>

<p align="center"><em>Table 2. Transfer accuracy and pretraining cost.</em></p>

These are transfer results following large-scale pretraining. Reported pretraining costs are lower than those of the comparison models, although training schedules and other settings also differ.

### 4.3 Pre-training Data Requirements

ViT trails CNNs with smaller pretraining datasets but improves strongly with scale. Larger ViT models also require more data to realize their advantage.

<p align="center">
  <img src="./images/figure3.jpg" alt="ImageNet transfer accuracy versus pretraining dataset size" width="500">
</p>

<p align="center"><em>Figure 3. Larger pretraining datasets benefit ViT.</em></p>

The paper also compares subsets of JFT, supporting the conclusion that convolutional inductive bias helps with limited data, while sufficiently large datasets let ViT learn useful spatial relationships.

### 4.4 Scaling Study

In a controlled comparison on JFT-300M, ViT needs approximately **2–4 times less pretraining compute** for comparable transfer performance averaged over five datasets. Hybrids help at smaller computational budgets, but their advantage diminishes at larger scales.

<p align="center">
  <img src="./images/figure5.jpg" alt="Transfer performance versus pretraining compute for ViT, ResNets, and hybrids" width="650">
</p>

<p align="center"><em>Figure 5. Transfer performance versus pretraining compute.</em></p>

### 4.5 Inspecting Vision Transformer

Learned position embeddings reflect spatial relationships between patches. Attention includes both local and global patterns, with attention distance generally increasing with depth.

<p align="center">
  <img src="./images/figure6.jpg" alt="Output-token attention projected onto input images" width="300">
</p>

<p align="center"><em>Figure 6. Output-token attention over image regions.</em></p>

Attention highlights regions relevant to classification, although attention weights alone do not establish causal explanations.

### 4.6 Self-supervision

The paper briefly explores masked patch prediction. It improves over training from scratch but remains below supervised pretraining in the reported experiment.

## 5 Conclusion

A standard Transformer can perform image classification effectively by treating patches as tokens. The central result is that large-scale pretraining makes this simple architecture competitive with strong CNNs; its gains depend on the data and compute available.
