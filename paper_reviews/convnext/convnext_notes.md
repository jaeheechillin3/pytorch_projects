# [Paper Review] A ConvNet for the 2020s

> **Authors:** Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie\
> **Institutions:** Facebook AI Research (FAIR), University of California, Berkeley\
> **Publication:** CVPR 2022\
> **Link:** [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

ConvNeXt modernizes a standard ResNet using design choices associated with vision Transformers, producing a convolution-only backbone competitive with hierarchical Transformers.

## 1 Introduction

Swin combines Transformer blocks with CNN-like locality and hierarchy. This motivates asking how much of its advantage comes from attention itself, rather than modern training and architectural choices.

## 2 Modernizing a ConvNet: a Roadmap

The paper progressively changes a ResNet-50 baseline. The cumulative ablations describe a particular design path; individual gains should not be treated as independent or universally additive.

### 2.1 Training Techniques

First strengthen training: use AdamW, a longer schedule, and modern augmentation and regularization. This improves the baseline before changing its architecture.

### 2.2 Macro Design

Redistribute blocks toward the third stage, using a $(3,3,9,3)$ stage-depth pattern in the ResNet-50-based exploration. Replace the overlapping convolution/pooling stem with a $4\times4$, stride-four patchifying convolution.

### 2.3 ResNeXt-ify

Use depthwise convolutions, separating spatial mixing within channels from pointwise channel mixing. Adjust width to maintain competitive capacity.

### 2.4 Inverted Bottleneck

Expand channels in the block's intermediate representation, resembling the expansion used by Transformer MLPs.

### 2.5 Large Kernel Sizes

Move depthwise spatial mixing before the pointwise expansion and increase the kernel size. A $7\times7$ kernel provides a favorable operating point in the studied design.

### 2.6 Micro Design

Use GELU, fewer activation and normalization layers, LayerNorm instead of BatchNorm, and separate downsampling layers. The final residual block combines depthwise spatial mixing with a channel-expanding and channel-contracting MLP.

```math
x_{\mathrm{out}}
=x+\operatorname{PW}_{\mathrm{contract}}
\left(\operatorname{GELU}
\left(\operatorname{PW}_{\mathrm{expand}}
\left(\operatorname{LN}(\operatorname{DWConv}_{7\times7}(x))\right)\right)\right).
```

This schematic omits layer scaling and stochastic depth.

## 3 Empirical Evaluations on ImageNet

### 3.1 Settings

Evaluate multiple model sizes under ImageNet-1K training and ImageNet-22K pretraining followed by fine-tuning. Match data and training conditions when comparing backbones.

### 3.2 Results

ConvNeXt competes favorably with Swin across model scales. Larger datasets and higher-resolution fine-tuning affect headline results, so retain those conditions when recording accuracy.

### 3.3 Isotropic ConvNeXt vs. ViT

Test constant-resolution convolutional models to separate the contribution of convolutional blocks from that of a hierarchical architecture.

## 4 Empirical Evaluation on Downstream Tasks

**COCO detection and instance segmentation.** Plug ConvNeXt into established detection frameworks and compare against other backbones.

**ADE20K semantic segmentation.** Evaluate dense prediction using a segmentation head.

**Efficiency.** FLOPs alone do not determine hardware throughput; operation types and implementation matter.

## 5 Related Work

Connect ConvNeXt to grouped/depthwise convolutions, inverted bottlenecks, and hybrid vision architectures.

## 6 Conclusions

The experiments challenge the assumption that attention is necessary for competitive visual backbones. Training and convolutional design remain substantial sources of improvement.

## Appendix notes

The supplements document training, robustness, detailed architectures, hardware benchmarks, and limitations.
