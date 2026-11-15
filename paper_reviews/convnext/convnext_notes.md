# [Paper Review] A ConvNet for the 2020s

> **Authors:** Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie\
> **Institutions:** Facebook AI Research (FAIR), University of California, Berkeley\
> **Publication:** CVPR 2022\
> **Link:** [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)


## Abstract & Introduction

Vision Transformers had begun to outperform conventional CNNs on image classification and dense prediction. However, these comparisons also involved different training procedures and architectural choices. Swin, for example, introduces locality and a feature hierarchy that resemble established CNN designs.

This paper asks how far a standard ConvNet can go when updated with those choices. Starting from ResNet-50, the authors progressively change its training and architecture to produce ConvNeXt. The final network uses convolution throughout and competes with hierarchical Transformers on classification, detection, and segmentation.

## Methods

### Training techniques

The first change strengthens the training recipe before modifying the network. Training runs for 300 epochs using AdamW, modern augmentation, and regularization, including Mixup, CutMix, RandAugment, random erasing, and stochastic depth.

This raises the ResNet-50 baseline from 76.1% to 78.8% ImageNet top-1 accuracy. A substantial part of the initial difference therefore comes from training.

### Macro design

The stage-depth ratio is changed from ResNet-50's $(3,4,6,3)$ to $(3,3,9,3)$, allocating more blocks to the third stage. The original convolution and max-pooling stem is replaced with a $4\times4$ stride-four convolution. This produces non-overlapping patch-like features while retaining a convolutional implementation.

### Depthwise convolutions and inverted bottlenecks

The spatial convolution becomes a depthwise convolution, applying one spatial filter per channel. Channel mixing is handled separately by pointwise transformations. The network width is adjusted to recover capacity.

The residual block also adopts an inverted bottleneck. Its intermediate representation expands to four times the input width before contracting again, resembling the channel expansion in a Transformer's MLP.

### Larger kernels

The depthwise convolution moves before the channel expansion, where it is cheaper to apply. Its kernel grows from $3\times3$ to $7\times7$, increasing the region mixed by each block. Increasing the kernel further gives little additional benefit in the reported experiment.

### Micro design

ReLU is replaced with GELU. Extra activation and normalization layers are removed, leaving a block with one activation between its pointwise transformations. Layer normalization replaces batch normalization, and separate downsampling layers connect the stages.

A simplified block is

```math
x_{\mathrm{out}}
=x+\operatorname{PW}_{\mathrm{contract}}\left(
\operatorname{GELU}\left(
\operatorname{PW}_{\mathrm{expand}}\left(
\operatorname{LN}(\operatorname{DWConv}_{7\times7}(x))
\right)\right)\right).
```

The implementation also includes layer scaling and stochastic depth. These changes are evaluated along a cumulative modernization path, so each measured gain depends on the modifications already made.

### Experiments

ConvNeXt-T, S, B, L, and XL vary stage depths and channel widths. Classification experiments use ImageNet-1K, with an additional setting that pretrains on ImageNet-22K. Downstream experiments replace the backbone in established detection and segmentation frameworks.

## Results

### ImageNet classification

At $224\times224$ resolution with ImageNet-1K training, ConvNeXt-T reaches **82.1% top-1 accuracy**, compared with Swin-T's 81.3%, at approximately the same FLOPs. ConvNeXt-B reaches **83.8%**, compared with Swin-B's 83.5%.

ImageNet-22K pretraining and higher-resolution fine-tuning improve larger models further. ConvNeXt-XL reaches **87.8%** at $384\times384$. This result uses more pretraining data and a larger model than the ImageNet-1K-only comparisons.

### Isotropic models

The paper also studies ConvNeXt variants that preserve one spatial resolution throughout the network, as ViT does. Their competitive results indicate that the updated convolutional block is useful beyond the hierarchical version.

### Detection and segmentation

ConvNeXt transfers well to COCO object detection and instance segmentation, and to ADE20K semantic segmentation. These experiments use the backbone's intermediate feature maps at several resolutions. Its gains therefore extend beyond image-level classification.

### Efficiency

ConvNeXt uses standard convolutional operations and requires no window partitioning or relative-position interpolation. The measured speed advantage depends on the hardware and implementation. The A100 experiments show a larger throughput advantage over Swin than the V100 experiments, despite similar theoretical FLOPs.

The results show that a carefully updated ConvNet remains a strong visual backbone. Modern training and architecture choices account for much of the progress that had accompanied the adoption of Transformers.
