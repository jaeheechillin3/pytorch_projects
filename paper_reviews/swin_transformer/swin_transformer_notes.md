# [Paper Review] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

> **Authors:** Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo\
> **Institutions:** Microsoft Research Asia\
> **Publication:** ICCV 2021\
> **Link:** [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

Swin builds a hierarchical vision backbone using local self-attention and shifted window partitions. It targets classification, detection, and segmentation within one architecture.

## 1 Introduction

Vanilla ViT maintains a single feature-map scale and uses global attention. Dense vision tasks require finer spatial information and efficient processing of high-resolution images. Swin introduces a feature hierarchy while restricting attention to fixed-size windows. Patch merging handles changes in spatial scale; shifted windows connect local attention regions. These solve different problems and are both needed for a general-purpose backbone.

## 2 Related Work

The paper compares CNN hierarchies, local self-attention, CNN–attention hybrids, and ViT-style backbones. Its key combination is multiscale features with efficient cross-window communication. CNNs provide a useful precedent: shallow layers preserve fine detail, while deeper layers summarize larger regions. Swin adopts this hierarchy without replacing attention with a fixed convolutional kernel. Its attention weights still depend on the image content.

## 3 Method

### 3.1 Overall Architecture

Split RGB images into $4\times4$ patches and linearly embed them. Four stages produce feature maps at resolutions $H/4$, $H/8$, $H/16$, and $H/32$. Patch merging concatenates each $2\times2$ group, then projects $4C$ channels into $2C$.

<p align="center">
  <!-- TODO: Add ./images/figure3.png, then uncomment the image below. -->
  <!-- <img src="./images/figure3.png" alt="Swin-T hierarchy and two successive Transformer blocks." width="600"> -->
</p>

<p align="center"><em>Figure 3. Swin-T hierarchy and two successive Transformer blocks. [Image to add]</em></p>

Each block uses pre-normalization, attention, and a two-layer GELU MLP, with residual connections around both sublayers. Merging reduces token count by four while doubling channel width, so deeper stages trade spatial detail for richer features. A $224\times224$ input begins with a $56\times56$ token grid and ends with a $7\times7$ grid. These intermediate outputs can feed a feature pyramid, allowing a downstream detector to combine fine and coarse spatial information.

### 3.2 Shifted Window based Self-Attention

**Window attention.** For a feature map of $h\times w$ tokens and windows of $M\times M$ tokens:

```math
\Omega(\mathrm{MSA})=4hwC^2+2(hw)^2C,
\qquad
\Omega(\mathrm{W\!-\!MSA})=4hwC^2+2M^2hwC.
```

With fixed $M$, window attention scales linearly with token count. This describes scaling with image area, not image side length. Global attention compares every token with every other token, producing a quadratic score matrix. Window attention instead computes many smaller matrices. The projection cost remains, so reducing attention interactions does not remove every source of computation.

**Shifted windows.** Alternate regular windows and windows shifted by $\lfloor M/2\rfloor$. Neighboring blocks therefore exchange information across previous window boundaries. A regular-window block alone cannot mix tokens across its partition. Shifting changes which tokens share a window, expanding the effective receptive field over successive blocks.

<p align="center">
  <!-- TODO: Add ./images/figure2.png, then uncomment the image below. -->
  <!-- <img src="./images/figure2.png" alt="Regular and shifted partitions create cross-window connections." width="600"> -->
</p>

<p align="center"><em>Figure 2. Regular and shifted partitions create cross-window connections. [Image to add]</em></p>

**Efficient computation.** Implement the shift cyclically, then mask artificial wraparound connections. This preserves the number of batched windows. Without the mask, the cyclic roll would introduce attention between originally distant image boundaries. The mask blocks these artificial connections rather than masking future tokens as in autoregressive attention.

<p align="center">
  <!-- TODO: Add ./images/figure4.png, then uncomment the image below. -->
  <!-- <img src="./images/figure4.png" alt="Cyclic shifting and attention masking for efficient batching." width="600"> -->
</p>

<p align="center"><em>Figure 4. Cyclic shifting and attention masking for efficient batching. [Image to add]</em></p>

**Relative position bias.** Add a learned bias based on relative token offsets:

```math
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}(QK^\top/\sqrt d+B)V.
```

Within an $M\times M$ window, each coordinate offset ranges from $-M+1$ to $M-1$. A compact table of relative offsets supplies the pairwise biases. This encodes spatial relationships while the query–key term continues to measure content similarity. The bias is therefore complementary to attention rather than a replacement for it.

### 3.3 Architecture Variants

Swin-T, S, B, and L vary channel width and stage depth. Swin-T uses initial width $C=96$ and stage depths $(2,2,6,2)$; Swin-S increases the third-stage depth to 18. The default window width is seven tokens. Compare model size, FLOPs, and measured throughput separately. Classification uses the final-stage representation after pooling, whereas dense prediction uses outputs from multiple stages. The same backbone thus supports tasks with different output structures.

## 4 Experiments

### 4.1 Image Classification on ImageNet-1K

Distinguish training on ImageNet-1K from ImageNet-22K pretraining. Swin-T reaches 81.3% top-1 at $224^2$; the headline 87.3% uses the larger Swin-L with additional pretraining. The standard ImageNet-1K recipe uses AdamW for 300 epochs. Resolution and pretraining data must match before interpreting accuracy or throughput differences. The comparisons also illustrate why the headline result cannot describe every model in the family: Swin-T and Swin-L differ substantially in capacity, training data, and evaluation resolution.

<p align="center">
  <!-- TODO: Add ./images/table1.png, then uncomment the image below. -->
  <!-- <img src="./images/table1.png" alt="ImageNet accuracy, parameters, FLOPs, and throughput across backbones." width="650"> -->
</p>

<p align="center"><em>Table 1. ImageNet accuracy, parameters, FLOPs, and throughput across backbones. [Image to add]</em></p>

### 4.2 Object Detection on COCO

Replace detector backbones with Swin. Evaluate both bounding-box and instance-mask AP rather than attributing complete detector results to the backbone alone. With Cascade Mask R-CNN, Swin-T reaches 50.5 box AP versus 46.3 for ResNet-50 under the compared setup. Keeping the detector and training recipe fixed makes this comparison more informative about backbone quality than comparing unrelated complete detection systems.

### 4.3 Semantic Segmentation on ADE20K

Use the hierarchy in a segmentation framework. The experiments test whether classification gains transfer to dense prediction. UperNet with Swin-S reaches 49.3 mIoU, while the larger ImageNet-22K-pretrained Swin-L reaches 53.5.

### 4.4 Ablation Study

Removing window shifts reduces performance. For Swin-T, shifting improves ImageNet top-1 from 80.2% to 81.3% and COCO box AP from 47.7 to 50.5. Relative position bias also helps. These controlled comparisons isolate the proposed mechanisms more directly than the headline benchmark scores. In particular, the shift ablation tests whether cross-window communication matters while retaining local attention. Compare attention variants using measured latency as well as theoretical computation.

<p align="center">
  <!-- TODO: Add ./images/table4.png, then uncomment the image below. -->
  <!-- <img src="./images/table4.png" alt="Ablations of window shifting and positional information." width="650"> -->
</p>

<p align="center"><em>Table 4. Ablations of window shifting and positional information. [Image to add]</em></p>

## 5 Conclusion

A Transformer can recover useful spatial hierarchy and locality while retaining content-dependent attention.

## Appendix notes

Detailed architectures, training recipes, resolution experiments, and the Swin MLP variant expand the main comparisons.
