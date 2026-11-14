# [Paper Review] Masked Autoencoders Are Scalable Vision Learners

> **Authors:** Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick\
> **Institutions:** Facebook AI Research (FAIR)\
> **Publication:** CVPR 2022\
> **Link:** [https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

MAE learns visual representations by reconstructing randomly hidden image patches. An asymmetric architecture makes heavy masking computationally useful as well as statistically challenging.

## 1 Introduction

Images contain substantial spatial redundancy. Reconstructing a small missing region may require little semantic understanding; removing most patches creates a harder learning task.

## 2 Related Work

The paper connects masked language modeling, denoising autoencoders, masked image prediction, and other self-supervised representation-learning methods.

## 3 Approach

**Masking.** Randomly remove a large fraction of image patches, typically 75%. The remaining patches provide a sparse view of the image.

**MAE encoder.** Feed only visible patch embeddings and their positional information into a ViT. No mask tokens enter the encoder, reducing its sequence length.

**MAE decoder.** Add learned mask tokens to the encoded visible tokens, restore spatial order, and use a lightweight decoder to predict pixels. The encoder and decoder need not have the same width or depth.

**Reconstruction target.** Minimize mean squared error over masked patches only:

```math
\mathcal L_{\mathrm{MAE}}
=\frac{1}{|M|}\sum_{i\in M}
\|\widehat x_i-x_i\|_2^2.
```

The paper also tests normalizing pixel values within each target patch. The decoder is discarded for downstream recognition.

**Implementation.** Randomly shuffle patch tokens, retain a subset for encoding, and use inverse shuffling to restore the full sequence.

## 4 ImageNet Experiments

### 4.1 Main Properties

**Masking ratio.** Heavy masking works well despite removing most input pixels.

**Decoder design.** A lightweight decoder can support useful encoder representations; reconstruction and recognition are distinct objectives.

**Mask tokens.** Inserting them only after encoding saves computation.

**Targets and augmentation.** Study raw versus normalized pixel targets, data augmentation, random versus structured masking, and training duration.

### 4.2 Comparisons with Previous Results

Compare full fine-tuning with supervised and self-supervised pretraining. ViT-Huge reaches 87.8% ImageNet-1K top-1 after fine-tuning using only ImageNet-1K pretraining data.

### 4.3 Partial Fine-tuning

Fine-tuning only some blocks can reveal representation quality that linear probing misses. A frozen linear classifier is therefore not the only useful evaluation.

## 5 Transfer Learning Experiments

Evaluate COCO detection/instance segmentation, ADE20K semantic segmentation, and additional classification datasets. Compare pixel reconstruction with discrete-token targets.

These experiments ask whether the representation transfers beyond the image-reconstruction task used in pretraining.

## 6 Discussion and Conclusion

A simple masked reconstruction objective can train large visual encoders effectively. Good representations do not require the downstream task to resemble reconstruction.

## Appendix notes

Training recipes, linear/partial probing, transfer settings, and robustness evaluation provide implementation and comparison details.
