# [Paper Review] Emerging Properties in Self-Supervised Vision Transformers

> **Authors:** Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin\
> **Institutions:** Facebook AI Research, Inria, Sorbonne University, Université Grenoble Alpes, CNRS, Grenoble INP\
> **Publication:** ICCV 2021\
> **Link:** [https://arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

DINO learns visual representations through self-distillation without labels. Self-supervised ViTs show strong nearest-neighbor classification and attention maps that often expose object structure.

## 1 Introduction

Image-level supervision compresses an image into a category label. The paper asks whether self-supervision can produce richer ViT features and reveal properties less apparent after supervised training.

## 2 Related Work

Relate DINO to contrastive learning, clustering, BYOL-style self-supervision, and knowledge distillation. Its teacher is built during training rather than supplied as a fixed, pretrained model.

## 3 Approach

### 3.1 SSL with Knowledge Distillation

Generate two global crops and several smaller local crops. The teacher processes global views; the student processes all views. Match distributions from different views of the same image:

```math
\mathcal L
=\sum_{v\in V_{\mathrm{global}}}
\sum_{\substack{v'\in V\\v'\ne v}}
H\!\left(\operatorname{sg}[P_t(v)],P_s(v')\right).
```

Here $H(p,q)=-\sum_k p_k\log q_k$, and $\operatorname{sg}$ stops teacher gradients.

**Teacher update.** Update teacher weights using an exponential moving average of student weights:

```math
\theta_t\leftarrow m\theta_t+(1-m)\theta_s.
```

**Avoiding collapse.** Center teacher logits using a running mean and sharpen the resulting distribution with temperature scaling. Centering and sharpening counter different degenerate solutions; agreement between networks alone is insufficient.

**Architecture.** Both branches contain a backbone and a projection head. Output dimensions represent learned assignments, not ground-truth class labels.

### 3.2 Implementation and Evaluation Protocols

Study both ResNets and ViTs. Evaluate frozen representations with weighted $k$-nearest neighbors and linear classification, alongside downstream transfer.

## 4 Main Results

### 4.1 Comparing with SSL Frameworks on ImageNet

Compare methods with matched backbones as well as across architectures. The reported ViT-Base linear evaluation reaches 80.1%; this is distinct from the 78.3% nearest-neighbor result for a small-patch ViT-S.

### 4.2 Properties of ViT Trained with SSL

#### 4.2.1 Nearest Neighbor Retrieval with DINO ViT

Study image retrieval and copy detection using frozen features.

#### 4.2.2 Discovering the Semantic Layout of Scenes

Inspect final-layer class-token attention and test video object segmentation through feature correspondences. Visually coherent attention maps are an empirical observation, not proof of causal explanations.

#### 4.2.3 Transfer Learning on Downstream Tasks

Evaluate whether pretrained features remain useful beyond ImageNet classification.

## 5 Ablation Study of DINO

### 5.1 Importance of the Different Components

Study multi-crop training, the momentum teacher, and patch size.

### 5.2 Impact of the Choice of Teacher Network

Compare teacher construction and training dynamics.

### 5.3 Avoiding Collapse

Investigate the interaction between centering and sharpening.

### 5.4 Compute Requirements

Compare representation quality under different computational budgets.

### 5.5 Training with Small Batches

Examine whether the method remains practical without contrastive learning's very large batches.

## 6 Conclusion

The pretraining objective changes the properties of a ViT's learned features. Self-distillation provides a route to useful representations without explicit negative pairs.

## Appendix notes

Projection-head design, teacher temperatures, crop settings, and evaluation recipes expand the main ablations.
