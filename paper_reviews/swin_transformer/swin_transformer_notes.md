# [Paper Review] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

> **Authors:** Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo\
> **Institutions:** Microsoft Research Asia\
> **Publication:** ICCV 2021\
> **Link:** [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)


## Abstract & Introduction

ViT applies a Transformer directly to image patches, but its single feature-map scale and global attention make it difficult to use for high-resolution dense prediction. Objects also vary greatly in size, requiring features that describe both fine detail and larger spatial regions.

This paper introduces the Swin Transformer, a hierarchical backbone for classification, object detection, and segmentation. It computes self-attention within local windows and shifts the window partition between consecutive blocks. This limits computation while allowing information to pass between neighboring windows.

## Methods

### Overall architecture

An RGB image is split into non-overlapping $4\times4$ patches. Each patch contains 48 values, which a linear layer projects into a $C$-dimensional embedding. Transformer blocks process this initial grid without changing its spatial resolution.

The network has four stages. Between stages, _patch merging_ concatenates each $2\times2$ group of neighboring tokens and projects the resulting $4C$ features into $2C$ dimensions. The feature resolutions are therefore $H/4$, $H/8$, $H/16$, and $H/32$, with increasing channel width.

<p align="center">
  <img src="https://arxiv.org/html/2103.14030v2/figs/teaser11.png" alt="Hierarchical Swin features compared with single-scale ViT features" width="600">
</p>

<p align="center"><em>Figure 1. Swin builds a feature hierarchy by merging neighboring patches.</em></p>

For a $224\times224$ image, the token grid begins at $56\times56$ and ends at $7\times7$. Detection and segmentation frameworks can use outputs from several stages. Classification instead applies global average pooling to the final-stage features, followed by a linear classifier.

### Window-based self-attention

Global attention compares every token with every other token. For an $h\times w$ token grid with channel width $C$, its cost is

```math
\Omega(\mathrm{MSA})=4hwC^2+2(hw)^2C.
```

Swin divides the grid into non-overlapping windows containing $M\times M$ tokens and computes attention independently within each window:

```math
\Omega(\mathrm{W\!-\!MSA})=4hwC^2+2M^2hwC.
```

For a fixed window size, the cost grows linearly with the number of image tokens. The default window contains $7\times7$ tokens. Restricting attention this way makes the early high-resolution stages practical.

### Shifted windows

A fixed partition would prevent communication between windows. Swin therefore alternates regular windows with windows shifted by $\lfloor M/2\rfloor$ tokens along each spatial axis.

<p align="center">
  <img src="https://arxiv.org/html/2103.14030v2/figs/teaser_v4.png" alt="Regular and shifted window partitions in successive layers" width="600">
</p>

<p align="center"><em>Figure 2. Shifting the partition connects neighboring windows across successive blocks.</em></p>

Tokens that belonged to separate windows can now attend to one another. Repeating this pattern lets information spread across the feature map.

The implementation performs a cyclic shift toward the top left before partitioning. Some resulting windows contain regions that were originally far apart at image boundaries. An attention mask blocks those artificial connections. This preserves the number of batched windows and avoids the overhead of separately padding many small boundary windows.

### Transformer blocks and relative position

Each block applies layer normalization before attention and before a two-layer GELU MLP. Both sublayers have residual connections:

```math
\widehat z^\ell=\operatorname{W\!-\!MSA}(\operatorname{LN}(z^{\ell-1}))+z^{\ell-1},
\qquad
z^\ell=\operatorname{MLP}(\operatorname{LN}(\widehat z^\ell))+\widehat z^\ell.
```

The next block uses shifted-window attention in the same structure. Within each attention head, a learned relative position bias is added to the query–key scores:

```math
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}(QK^\top/\sqrt d+B)V.
```

The bias depends on the relative horizontal and vertical offset of a pair of tokens. An $M\times M$ window requires a table covering $(2M-1)^2$ possible offsets per head.

### Model variants and training

Swin-T uses initial width 96 and stage depths $(2,2,6,2)$. Swin-S increases the third stage to 18 blocks. Swin-B and Swin-L increase width further.

ImageNet-1K training uses AdamW for 300 epochs. Larger models are also pretrained on ImageNet-22K, then fine-tuned for ImageNet-1K classification. The experiments additionally use Swin as the backbone in COCO detection and ADE20K segmentation frameworks.

## Results

### Image classification

Swin-T achieves **81.3% ImageNet top-1 accuracy** at $224\times224$, with 29 million parameters and 4.5 GFLOPs. Swin-S reaches 83.0%, and Swin-B reaches 83.5% at the same resolution with ImageNet-1K training.

The larger Swin-L reaches **87.3%** using ImageNet-22K pretraining and $384\times384$ fine-tuning. This is the paper's headline classification result.

### Object detection and segmentation

With Cascade Mask R-CNN, Swin-T reaches **50.5 box AP** on COCO, compared with 46.3 for ResNet-50 in the corresponding comparison. The hierarchy supplies feature maps at the resolutions needed by the detector.

For ADE20K semantic segmentation, UperNet with Swin-S reaches **49.3 mIoU**. ImageNet-22K-pretrained Swin-L reaches **53.5 mIoU**. These results show that the backbone transfers to tasks requiring predictions across the image.

### Ablation studies

Removing window shifts lowers Swin-T's ImageNet accuracy from 81.3% to 80.2% and COCO box AP from 50.5 to 47.7. Relative position bias also improves performance. The shift ablation demonstrates that communication across windows contributes beyond the efficiency of local attention alone.

The paper's [Tables 1–4](https://arxiv.org/pdf/2103.14030) contain the classification, dense-prediction, and ablation comparisons.
