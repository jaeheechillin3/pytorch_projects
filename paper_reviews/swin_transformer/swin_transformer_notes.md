# [Paper Review] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

> **Authors:** Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo\
> **Institutions:** Microsoft Research Asia\
> **Publication:** ICCV 2021\
> **Link:** [https://arxiv.org/abs/2103.14030](https://arxiv.org/abs/2103.14030)

## Abstract & Introduction

Images contain objects at different scales, and dense prediction needs more spatial detail than a single coarse token grid provides. Swin builds a Transformer feature hierarchy and restricts attention to local windows, shifting their boundaries between blocks to exchange information. This makes computation scale linearly with image area while supporting classification, detection, and segmentation with the same backbone.

## Methods

<p align="center">
  <img src="https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AtXCiVjcULy5wYvUptEntRA.png" alt="Figure 3. Swin-T architecture and two successive Swin Transformer blocks." width="700">
</p>

<p align="center"><em>Figure 3. Swin-T architecture and two successive Swin Transformer blocks. From the <a href="https://arxiv.org/abs/2103.14030">original paper</a>.</em></p>

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
\widehat z^\ell=\mathrm{W\!-\!MSA}(\mathrm{LN}(z^{\ell-1}))+z^{\ell-1},
\qquad
z^\ell=\mathrm{MLP}(\mathrm{LN}(\widehat z^\ell))+\widehat z^\ell.
```

The next block uses shifted-window attention in the same structure. Within each attention head, a learned relative position bias is added to the query–key scores:

```math
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}(QK^\top/\sqrt d+B)V.
```

The bias depends on the relative horizontal and vertical offset of a pair of tokens. An $M\times M$ window requires a table covering $(2M-1)^2$ possible offsets per head.

### Model variants and training

Swin-T uses initial width 96 and stage depths $(2,2,6,2)$. Swin-S increases the third stage to 18 blocks. Swin-B and Swin-L increase width further.

ImageNet-1K training uses AdamW for 300 epochs. Larger models are also pretrained on ImageNet-22K, then fine-tuned for ImageNet-1K classification. The experiments additionally use Swin as the backbone in COCO detection and ADE20K segmentation frameworks.

## Results

### ImageNet-1K classification

| Model | Pretraining | Input | GFLOPs | Top-1 ↑ |
| --- | --- | --- | ---: | ---: |
| Swin-T | ImageNet-1K | 224² | 4.5 | 81.3 |
| Swin-S | ImageNet-1K | 224² | 8.7 | 83.0 |
| Swin-B | ImageNet-1K | 224² | 15.4 | 83.5 |
| Swin-B | ImageNet-1K | 384² | 47.0 | 84.5 |
| Swin-L | ImageNet-22K | 384² | 103.9 | 87.3 |

_Selected results from Table 1._ Higher capacity, resolution, and additional pretraining data each contribute to the strongest result.

### Window-shift ablation

| Swin-T variant | ImageNet top-1 ↑ | COCO box AP ↑ | COCO mask AP ↑ | ADE20K mIoU ↑ |
| --- | ---: | ---: | ---: | ---: |
| No window shifts | 80.2 | 47.7 | 41.5 | 43.3 |
| Shifted windows | 81.3 | 50.5 | 43.7 | 46.1 |

_Selected results from Table 4._ Cross-window communication improves all three tasks with little latency overhead.

### Position-bias ablation

| Position information | ImageNet top-1 ↑ | COCO box AP ↑ | ADE20K mIoU ↑ |
| --- | ---: | ---: | ---: |
| None | 80.1 | 49.2 | 43.8 |
| Absolute | 80.5 | 49.0 | 43.2 |
| Relative | 81.3 | 50.5 | 46.1 |

_Selected results from Table 4._ Relative position bias helps both recognition and dense prediction, while absolute embeddings do not transfer as consistently.
