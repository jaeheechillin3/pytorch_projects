# [Paper Review] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

> **Authors:** Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo\
> **Institutions:** Microsoft Research Asia\
> **Publication:** ICCV 2021\
> **Link:** [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

Swin builds a hierarchical vision backbone using local self-attention and shifted window partitions. It targets classification, detection, and segmentation within one architecture.

## 1 Introduction

Vanilla ViT maintains a single feature-map scale and uses global attention. Dense vision tasks require finer spatial information and efficient processing of high-resolution images. Swin introduces a feature hierarchy while restricting attention to fixed-size windows.

## 2 Related Work

The paper compares CNN hierarchies, local self-attention, CNN–attention hybrids, and ViT-style backbones. Its key combination is multiscale features with efficient cross-window communication.

## 3 Method

### 3.1 Overall Architecture

Split RGB images into $4\times4$ patches and linearly embed them. Four stages produce feature maps at resolutions $H/4$, $H/8$, $H/16$, and $H/32$. Patch merging concatenates each $2\times2$ group, then projects $4C$ channels into $2C$.

Each block uses pre-normalization, attention, and a two-layer GELU MLP, with residual connections around both sublayers.

### 3.2 Shifted Window based Self-Attention

**Window attention.** For a feature map of $h\times w$ tokens and windows of $M\times M$ tokens:

```math
\Omega(\mathrm{MSA})=4hwC^2+2(hw)^2C,
\qquad
\Omega(\mathrm{W\!-\!MSA})=4hwC^2+2M^2hwC.
```

With fixed $M$, window attention scales linearly with token count. This describes scaling with image area, not image side length.

**Shifted windows.** Alternate regular windows and windows shifted by $\lfloor M/2\rfloor$. Neighboring blocks therefore exchange information across previous window boundaries.

**Efficient computation.** Implement the shift cyclically, then mask artificial wraparound connections. This preserves the number of batched windows.

**Relative position bias.** Add a learned bias based on relative token offsets:

```math
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}(QK^\top/\sqrt d+B)V.
```

### 3.3 Architecture Variants

Swin-T, S, B, and L vary channel width and stage depth. The default window width is seven tokens. Compare model size, FLOPs, and measured throughput separately.

## 4 Experiments

### 4.1 Image Classification on ImageNet-1K

Distinguish training on ImageNet-1K from ImageNet-22K pretraining. Swin-T reaches 81.3% top-1 at $224^2$; the headline 87.3% uses the larger Swin-L with additional pretraining.

### 4.2 Object Detection on COCO

Replace detector backbones with Swin. Evaluate both bounding-box and instance-mask AP rather than attributing complete detector results to the backbone alone.

### 4.3 Semantic Segmentation on ADE20K

Use the hierarchy in a segmentation framework. The experiments test whether classification gains transfer to dense prediction.

### 4.4 Ablation Study

Removing window shifts reduces performance. Relative position bias also helps. Compare attention variants using measured latency as well as theoretical computation.

## 5 Conclusion

A Transformer can recover useful spatial hierarchy and locality while retaining content-dependent attention.

## Appendix notes

Detailed architectures, training recipes, resolution experiments, and the Swin MLP variant expand the main comparisons.
