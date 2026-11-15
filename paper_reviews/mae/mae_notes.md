# [Paper Review] Masked Autoencoders Are Scalable Vision Learners

> **Authors:** Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick\
> **Institutions:** Facebook AI Research (FAIR)\
> **Publication:** CVPR 2022\
> **Link:** [https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)


## Abstract & Introduction

Masked prediction has been effective for language pretraining, as demonstrated by BERT. Applying the same idea to images introduces a different problem: nearby image patches are highly redundant, so predicting a small missing region may require little understanding of the scene.

This paper introduces masked autoencoders (MAEs), which reconstruct images after removing most of their patches. Two choices make this effective. A high masking ratio creates a difficult prediction task, and an asymmetric encoder–decoder architecture reduces the cost of processing the corrupted image.

The pretrained encoder is then used for recognition. The reconstruction decoder is discarded, allowing its design to focus on the pretraining task.

## Methods

### Random masking

The image is divided into non-overlapping patches as in ViT. A subset is sampled uniformly without replacement, and the remaining patches are removed. The default masking ratio is 75%.

For a $224\times224$ image with $16\times16$ patches, there are 196 image patches. A 75% mask leaves only 49 visible patches. Their spatial distribution provides scattered evidence about the scene while removing much of the local redundancy.

### Asymmetric encoder and decoder

<p align="center">
  <img src="https://arxiv.org/html/2111.06377v3/arch.png" alt="MAE encoder processes visible patches and a lightweight decoder reconstructs missing patches" width="600">
</p>

<p align="center"><em>Figure 1. Visible patches enter the encoder; mask tokens are introduced in the decoder.</em></p>

The encoder is a ViT applied only to the visible patches. Each patch receives a linear embedding and its positional embedding before entering the Transformer. The original positions are retained even though the encoder processes a shortened sequence.

After encoding, the visible representations are projected into the decoder's feature width. Learned mask tokens are inserted at the missing positions, and the full sequence is restored to its original spatial order. Positional embeddings distinguish the locations of otherwise identical mask tokens.

A small Transformer decoder predicts the pixels of each patch. The default decoder has eight blocks and width 512, making it much smaller than a ViT-Large encoder. The expensive encoder therefore processes only a quarter of the image tokens, while the lightweight decoder handles the complete sequence.

### Reconstruction objective

The final decoder layer projects each token into a vector of patch pixels. Training minimizes mean squared error on the masked patches:

```math
\mathcal L_{\mathrm{MAE}}
=\frac{1}{|M|}\sum_{i\in M}
\frac{1}{P^2C}\left\|\widehat x_i-x_i\right\|_2^2.
```

Here $M$ is the set of masked patches, $P$ is patch width, and $C$ is the number of image channels. The patch vector contains $P^2C$ pixel values.

The paper also normalizes each target patch using its own mean and standard deviation. This improves representation quality. Only masked patches contribute to the loss, so the model is not directly trained to reconstruct the visible patches.

### Implementation

Random shuffling makes masking straightforward. The model shuffles the patch sequence and keeps its first quarter for encoding. It appends mask tokens after encoding, then applies the inverse permutation to restore the complete spatial arrangement.

This uses ordinary dense Transformer operations on a shorter sequence. No specialized sparse attention implementation is required.

### Pretraining and evaluation

Pretraining uses ImageNet-1K images without labels. Evaluation includes full fine-tuning, linear probing with a frozen encoder, and partial fine-tuning of selected encoder blocks. Downstream recognition uses the complete unmasked image.

The paper studies ViT-Base, Large, and Huge. The ablations use ViT-Large to examine masking, decoder design, target normalization, augmentation, and pretraining duration.

## Results

<p align="center">
  <img src="https://arxiv.org/html/2111.06377v3/samples.png" alt="Masked inputs, MAE reconstructions, and original images" width="600">
</p>

<p align="center"><em>Figure 2. Masked images, reconstructions, and targets at an 80% masking ratio.</em></p>

### Masking ratio

A masking ratio of 75% works well for both full fine-tuning and linear probing. This is substantially higher than the proportion of tokens selected for prediction in BERT. Heavy masking removes enough visual redundancy to make the reconstruction task useful.

Random masking also performs better than the studied structured alternatives. Leaving scattered visible patches preserves more distributed evidence than removing large contiguous regions at the same masking ratio.

### Decoder and mask tokens

Decoder depth affects linear-probing performance more strongly than full fine-tuning. With an eight-block decoder, the ViT-L baseline reaches 84.9% after fine-tuning and 73.5% under linear probing. A single-block decoder gives similar fine-tuning accuracy but a much lower linear-probing score.

Excluding mask tokens from the encoder reduces training FLOPs by about **3.3 times** and measured training time by about **2.8 times** in the reported ablation. It also improves accuracy, making the asymmetric architecture useful for both efficiency and representation learning.

### Targets, augmentation, and training duration

Normalized pixel targets improve the baseline fine-tuning result from 84.9% to 85.4%. They also perform competitively with discrete visual tokens, without requiring a separately trained image tokenizer.

Random cropping alone provides effective augmentation. Strong color transformations, which are useful in contrastive learning, provide little benefit here. Longer pretraining improves the representations, with 1,600-epoch runs producing strong final results.

### ImageNet classification

With ImageNet-1K-only pretraining, ViT-Huge reaches **86.9% top-1 accuracy** when fine-tuned at $224\times224$, and **87.8%** at $448\times448$. Larger encoders benefit from masked pretraining, even though training the same large models from scratch can overfit.

Partial fine-tuning improves substantially over a linear probe. The representations can therefore contain useful information that becomes accessible after adapting only part of the encoder.

### Transfer learning

MAE pretraining improves COCO object detection and instance segmentation, as well as ADE20K semantic segmentation, over the corresponding supervised-pretraining baselines. The gains grow with larger encoders. These experiments show that predicting missing pixels can produce features useful for recognition and localization.

The paper's [Figure 5 and Tables 1–5](https://arxiv.org/pdf/2111.06377) present the masking ablations, ImageNet comparisons, and transfer results.
