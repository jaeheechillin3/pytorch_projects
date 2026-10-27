# [Paper Review] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

> **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby\
> **Institutions:** Google Research, Brain Team\
> **Publication:** ICLR 2021\
> **Link:** https://arxiv.org/abs/2010.11929

## Abstract & Introduction

In natural language processing (NLP), Transformers have become the dominant architecture. Through pretraining on a large text corpus then fine-tuning, Transformers have shown unprecedented performance on all NLP tasks. However, convolutional neural networks (CNNs) remain the dominant architecture in computer vision. There have been multiple attempts to introduce the self-attention mechanism to various parts of CNNs, though none have been particularly successful. 

This paper takes a different approach; instead of relying on CNNs, Transformers are applied directly to supervised image classification with as few modifications from the original architecture as possible. Images are split into patches, each of which is treated the same way as tokens in NLP tasks. 

On smaller datasets, Vision Transformers (ViTs) slightly lag behind in accuracy compared to CNNs of comparable sizes. This is due to ViTs lacking inductive biases like translation equivariance and locality, both of which are inherent to CNNs. However, ViTs perform at the state of the art level on large datasets. 

## Methods

<p align="center">
  <img src="images/figure1.jpg" width="600">
</p>

<p align="center">
  *Figure 1. The generator maps a noise vector to a 64 × 64 image.*
</p>



<p align="center">
  <img src="images/figure3.jpg" width="400">
</p>

<p align="center">
  *Figure 1. The generator maps a noise vector to a 64 × 64 image.*
</p>



<p align="center">
  <img src="images/figure5.jpg" width="600">
</p>

<p align="center">
  *Figure 1. The generator maps a noise vector to a 64 × 64 image.*
</p>



<p align="center">
  <img src="images/figure6.jpg" width="150">
</p>

<p align="center">
  *Figure 1. The generator maps a noise vector to a 64 × 64 image.*
</p>



<p align="center">
  <img src="images/table2.jpeg" width="600">
</p>

<p align="center">
  *Figure 1. The generator maps a noise vector to a 64 × 64 image.*
</p>






3


4.1 
## Results

4.2 

4.3 

4.4

4.5

4.6 
