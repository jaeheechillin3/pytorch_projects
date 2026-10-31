# [Paper Review] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

> **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby\
> **Institutions:** Google Research, Brain Team\
> **Publication:** ICLR 2021\
> **Link:** https://arxiv.org/abs/2010.11929

## Abstract & Introduction

Vision Transformer applies a Transformer encoder to image patches. It tests whether pretraining compensates for weaker inductive biases than CNNs.

## Methods

An image of size $H\times W$ is split into $N=HW/P^2$ non-overlapping patches. Flattened patches are projected into embeddings; a learnable class token and position embeddings form the input:

$$
z_0=[x_{\mathrm{class}};x_p^1E;\ldots;x_p^NE]+E_{\mathrm{pos}}.
$$

Pre-normalized attention and MLP blocks use residual connections. The final class-token representation feeds a classification head. 


<p align="center">
  <img src="images/figure1.jpg" alt="Vision Transformer patch embedding and encoder architecture" width="600">
</p>

<p align="center"><em>Figure 1. Patch embeddings enter a Transformer encoder.</em></p>

Models are pretrained on ImageNet, ImageNet-21k, or JFT-300M, then fine-tuned; position embeddings are interpolated for higher resolution.

## Results

### Transfer accuracy

JFT-pretrained **ViT-H/14 reaches 88.55% ImageNet and 94.55% CIFAR-100 accuracy** after fine-tuning (Table 2).

<p align="center">
  <img src="images/table2.jpeg" alt="Fine-tuning accuracy and pretraining cost across image classification benchmarks" width="600">
</p>

<p align="center"><em>Table 2. Transfer accuracy and pretraining cost.</em></p>

### Data and compute

ViT trails CNNs with smaller pretraining datasets but improves strongly with scale. In the controlled scaling study, it needs approximately **2–4 times less pretraining compute** for comparable average transfer performance.

<p align="center">
  <img src="images/figure3.jpg" alt="ImageNet transfer accuracy versus pretraining dataset size" width="500">
</p>

<p align="center"><em>Figure 3. Larger pretraining datasets benefit ViT.</em></p>

<p align="center">
  <img src="images/figure5.jpg" alt="Transfer performance versus pretraining computation for ViT ResNets and hybrids" width="600">
</p>

<p align="center"><em>Figure 5. Performance versus pretraining compute.</em></p>

### Attention visualization

Class-token attention highlights relevant image regions. Attention weights alone do not establish causal explanations.

<p align="center">
  <img src="images/figure6.jpg" alt="Class-token attention projected onto input images" width="400">
</p>

<p align="center"><em>Figure 6. Output-token attention over image regions.</em></p>

The gains depend on substantial pretraining data.
