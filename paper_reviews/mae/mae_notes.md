# [Paper Review] Masked Autoencoders Are Scalable Vision Learners

> **Authors:** Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick\
> **Institutions:** Facebook AI Research (FAIR)\
> **Publication:** CVPR 2022\
> **Link:** [https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

MAE learns visual representations by reconstructing randomly hidden image patches. An asymmetric architecture makes heavy masking computationally useful as well as statistically challenging.

## 1 Introduction

Images contain substantial spatial redundancy. Reconstructing a small missing region may require little semantic understanding; removing most patches creates a harder learning task. Unlike BERT's sparse token masking, MAE uses heavy masking to reduce visual redundancy. The goal is a transferable encoder, rather than a reconstruction system alone.

## 2 Related Work

The paper connects masked language modeling, denoising autoencoders, masked image prediction, and other self-supervised representation-learning methods. Contrastive methods learn agreement between augmented views, whereas MAE predicts missing content. The difference changes what must be preserved in the representation and which augmentations are useful. Masking itself supplies a stochastic prediction problem each time an image is sampled.

## 3 Approach

<p align="center">
  <!-- TODO: Add ./images/figure1.png, then uncomment the image below. -->
  <!-- <img src="./images/figure1.png" alt="Visible patches enter the encoder; mask tokens enter only the decoder." width="600"> -->
</p>

<p align="center"><em>Figure 1. Visible patches enter the encoder; mask tokens enter only the decoder. [Image to add]</em></p>

**Masking.** Randomly remove a large fraction of image patches, typically 75%. The remaining patches provide a sparse view of the image. For a $224\times224$ image with $16\times16$ patches, 196 tokens become 49 visible tokens at a 75% masking ratio. Sampling visible patches uniformly without replacement distributes evidence across the image. Heavy random masking makes simple interpolation from immediately adjacent pixels less reliable.

**MAE encoder.** Feed only visible patch embeddings and their positional information into a ViT. No mask tokens enter the encoder, reducing its sequence length. Positional embeddings preserve the original locations of visible patches, even though those patches are processed as a shorter sequence.

**MAE decoder.** Add learned mask tokens to the encoded visible tokens, restore spatial order, and use a lightweight decoder to predict pixels. The encoder and decoder need not have the same width or depth. The default ablation decoder has eight blocks and width 512; it handles reconstruction-specific processing while the larger encoder learns transferable features. The decoder processes the complete token sequence, but its smaller width and depth keep that stage relatively inexpensive. A shared learned mask token initially marks every missing location; positional embeddings distinguish where each missing patch belongs.

**Reconstruction target.** Minimize mean squared error over masked patches only:

```math
\mathcal L_{\mathrm{MAE}}
=\frac{1}{|M|}\sum_{i\in M}
\|\widehat x_i-x_i\|_2^2.
```

The paper also tests normalizing pixel values within each target patch. The decoder is discarded for downstream recognition. Normalization subtracts each target patch's mean and divides by its standard deviation. This changes the prediction target, rather than normalizing the whole input image. Because visible patches do not contribute to the loss, their predicted pixels need not look as good as the hidden-region predictions. Reconstruction figures should therefore be interpreted with the loss mask in mind. Plausible completion also need not exactly recover an inherently ambiguous hidden region.

**Implementation.** Randomly shuffle patch tokens, retain a subset for encoding, and use inverse shuffling to restore the full sequence.

<p align="center">
  <!-- TODO: Add ./images/figure2.png, then uncomment the image below. -->
  <!-- <img src="./images/figure2.png" alt="ImageNet inputs, MAE reconstructions, and targets at 80% masking." width="600"> -->
</p>

<p align="center"><em>Figure 2. ImageNet inputs, MAE reconstructions, and targets at 80% masking. [Image to add]</em></p>

## 4 ImageNet Experiments

### 4.1 Main Properties

**Masking ratio.** Heavy masking works well despite removing most input pixels. Fine-tuning is less sensitive to the ratio than linear probing, so reconstruction loss alone does not identify the best representation.

<p align="center">
  <!-- TODO: Add ./images/figure5.png, then uncomment the image below. -->
  <!-- <img src="./images/figure5.png" alt="Masking ratio versus fine-tuning and linear-probing accuracy." width="600"> -->
</p>

<p align="center"><em>Figure 5. Masking ratio versus fine-tuning and linear-probing accuracy. [Image to add]</em></p>

**Decoder design.** A lightweight decoder can support useful encoder representations; reconstruction and recognition are distinct objectives. A decoder with enough capacity can absorb some of the pixel-level specialization, allowing the encoder output to remain more useful for recognition. Decoder settings can affect linear probing more strongly than full fine-tuning.

**Mask tokens.** Inserting them only after encoding saves computation. In the ViT-L ablation, excluding encoder mask tokens reduces training FLOPs by about 3.3 times and measured training time by about 2.8 times.

**Targets and augmentation.** Study raw versus normalized pixel targets, data augmentation, random versus structured masking, and training duration. Cropping-only augmentation works well; the contrastive-learning recipe of strong color transformations is not automatically beneficial. Longer pretraining helps, partly because each epoch exposes the encoder to only a fraction of the patches. Structured block masking is also harder than random masking at the same high ratio, showing that the spatial arrangement of missing information matters.

<p align="center">
  <!-- TODO: Add ./images/table1.png, then uncomment the image below. -->
  <!-- <img src="./images/table1.png" alt="Decoder, mask-token, target, augmentation, and sampling ablations." width="650"> -->
</p>

<p align="center"><em>Table 1. Decoder, mask-token, target, augmentation, and sampling ablations. [Image to add]</em></p>

### 4.2 Comparisons with Previous Results

Compare full fine-tuning with supervised and self-supervised pretraining. ViT-Huge reaches 87.8% ImageNet-1K top-1 after fine-tuning using only ImageNet-1K pretraining data. That result uses $448\times448$ fine-tuning; the corresponding $224\times224$ result is 86.9%. Keep evaluation resolution separate from the pretraining dataset.

<p align="center">
  <!-- TODO: Add ./images/table3.png, then uncomment the image below. -->
  <!-- <img src="./images/table3.png" alt="End-to-end fine-tuning comparisons across ViT model sizes." width="650"> -->
</p>

<p align="center"><em>Table 3. End-to-end fine-tuning comparisons across ViT model sizes. [Image to add]</em></p>

### 4.3 Partial Fine-tuning

Fine-tuning only some blocks can reveal representation quality that linear probing misses. A frozen linear classifier is therefore not the only useful evaluation. Allowing nonlinear adaptation can substantially change the ranking of pretrained representations.

## 5 Transfer Learning Experiments

Evaluate COCO detection/instance segmentation, ADE20K semantic segmentation, and additional classification datasets. Compare pixel reconstruction with discrete-token targets.

These experiments ask whether the representation transfers beyond the image-reconstruction task used in pretraining. COCO evaluates localization and instance masks, while ADE20K tests semantic labels across image pixels. Improvements here provide evidence beyond a single classification benchmark. The comparisons also show that predicting normalized pixels can be competitive without requiring an additional pretrained image tokenizer.

## 6 Discussion and Conclusion

A simple masked reconstruction objective can train large visual encoders effectively. Good representations do not require the downstream task to resemble reconstruction. The main contribution combines the learning objective with an efficient asymmetric implementation; either idea alone gives an incomplete account of the method.

## Appendix notes

Training recipes, linear/partial probing, transfer settings, and robustness evaluation provide implementation and comparison details.
