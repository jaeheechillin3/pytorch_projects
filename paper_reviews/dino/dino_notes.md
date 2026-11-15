# [Paper Review] Emerging Properties in Self-Supervised Vision Transformers

> **Authors:** Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin\
> **Institutions:** Facebook AI Research, Inria, Sorbonne University, Université Grenoble Alpes, CNRS, Grenoble INP\
> **Publication:** ICCV 2021\
> **Link:** [https://arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)


## Abstract & Introduction

Vision Transformers can learn strong image representations with supervised pretraining, but the pretraining objective may affect what those representations contain. This paper examines the properties that emerge when a ViT is trained without labels.

The proposed method, DINO, learns through _self-distillation with no labels_. A student network predicts the output of a teacher network for another view of the same image. The teacher is updated from the student during training. The resulting features work well for nearest-neighbor classification, and the attention maps often separate objects from their backgrounds.

## Methods

### Student and teacher networks

Both networks contain an image backbone and a projection head. Their architectures are the same, but their parameters differ. Each produces a vector of logits, which a temperature softmax converts into a probability distribution:

```math
P_s(x)_k=
\frac{\exp(g_{\theta_s}(x)_k/\tau_s)}
{\sum_j\exp(g_{\theta_s}(x)_j/\tau_s)}.
```

The teacher distribution is computed similarly, with its own temperature and centered logits. These output dimensions are learned feature assignments; they do not correspond to annotated ImageNet categories.

For two views of an image, the student is trained to match the teacher using cross entropy. The full objective sums across different views:

```math
\mathcal L=
\sum_{v\in V_{\mathrm{global}}}
\sum_{\substack{v'\in V\\v'\ne v}}
H\!\left(\operatorname{sg}[P_t(v)],P_s(v')\right).
```

Here $H(p,q)=-\sum_k p_k\log q_k$, and stop-gradient prevents the teacher from receiving backpropagated updates.

### Momentum teacher

The teacher parameters follow an exponential moving average of the student parameters:

```math
\theta_t\leftarrow m\theta_t+(1-m)\theta_s.
```

The momentum increases toward one during training. This produces a slowly changing target that aggregates information from successive student models. The teacher is learned together with the student, so no separately pretrained teacher is required.

### Centering and sharpening

Agreement alone permits uninformative solutions, such as predicting the same distribution for every image. DINO combines two mechanisms to avoid this collapse.

Centering subtracts a running mean from the teacher logits, reducing the tendency for one output dimension to dominate. Sharpening uses a low teacher temperature to make the target distribution more concentrated. Centering alone encourages overly uniform predictions, while sharpening alone can favor one dominant dimension. Their combination balances these tendencies.

### Multi-crop training

Each image produces two large global crops and several smaller local crops. The teacher processes only the global views, while the student processes all views. Matching a local view to a teacher's global prediction encourages the student to relate image parts to a larger scene.

The same backbone handles the different crop sizes. Local crops provide extra training views at a lower cost than processing the same number of full-size images.

### Experiments

The paper evaluates both ResNets and ViTs pretrained on ImageNet without labels. Linear evaluation trains a classifier over frozen representations. Weighted nearest-neighbor evaluation instead predicts a label directly from nearby training-image features, with no fitted classifier.

## Results

### ImageNet classification

DINO reaches **80.1% top-1 accuracy** in linear evaluation with a small-patch ViT-Base. A small-patch ViT-S reaches **78.3%** using nearest-neighbor classification. The latter result shows that images with related semantics are already grouped usefully in the frozen feature space.

### Object structure and retrieval

<p align="center">
  <img src="https://arxiv.org/html/2104.14294v2/attn6.png" alt="Final-layer class-token attention maps from a self-supervised ViT" width="600">
</p>

<p align="center"><em>Figure 1. Class-token attention from a ViT trained without labels.</em></p>

Several attention heads highlight coherent foreground objects even though training provides no segmentation masks. The paper also propagates object labels through video using feature correspondences and evaluates image retrieval and copy detection. These experiments examine spatial and instance information beyond classification.

### Ablation studies

The momentum teacher and multi-crop training both improve representation quality. Smaller image patches also help substantially, although they lengthen the token sequence and increase computation.

The centering and sharpening experiments show that the teacher's output distribution is essential to stable learning. The method also remains effective with smaller batches, making it less dependent on the very large batches used to supply negatives in some contrastive approaches.

Together, the experiments show that self-supervision changes the features learned by a ViT. DINO produces useful global image representations while retaining spatial information visible in its attention maps.
