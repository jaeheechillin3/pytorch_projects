# [Paper Review] Emerging Properties in Self-Supervised Vision Transformers

> **Authors:** Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin\
> **Institutions:** Facebook AI Research, Inria, Sorbonne University, Université Grenoble Alpes, CNRS, Grenoble INP\
> **Publication:** ICCV 2021\
> **Link:** [https://arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)

## Abstract & Introduction

What does a Vision Transformer learn when no one tells it the image's class? DINO trains a student to match a momentum teacher across different crops, using centering and sharpening to keep the predictions informative. The resulting representations support strong nearest-neighbor classification and often reveal foreground objects in their attention maps, despite receiving no segmentation labels.

## Methods

<p align="center">
  <img src="https://i-blog.csdnimg.cn/blog_migrate/ad2824720a3c4071bc28c97081d5a247.png" alt="Figure 2. Self-distillation with no labels." width="700">
</p>

<p align="center"><em>Figure 2. Self-distillation with no labels. From the <a href="https://arxiv.org/abs/2104.14294">original paper</a>.</em></p>

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
H\!\left(\mathrm{sg}[P_t(v)],P_s(v')\right).
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

### ImageNet frozen-feature evaluation

| Method / backbone | Linear top-1 ↑ | k-NN top-1 ↑ |
| --- | ---: | ---: |
| BYOL / ResNet-50 | 74.4 | 64.8 |
| SwAV / ResNet-50 | 75.3 | 65.7 |
| DINO / ResNet-50 | 75.3 | 67.5 |
| DINO / ViT-S/16 | 77.0 | 74.5 |
| DINO / ViT-B/16 | 78.2 | 76.1 |
| DINO / ViT-S/8 | 79.7 | 78.3 |
| DINO / ViT-B/8 | 80.1 | 77.4 |

_Selected results from Table 2._ Smaller patches improve frozen features, and the strong k-NN results show useful semantic organization without fitting a classifier.

### Components and emergent properties

| Experiment | Reported finding |
| --- | --- |
| Momentum teacher | Improves the student's representation and supplies more stable targets |
| Multi-crop training | Improves performance by matching local views to global views |
| Centering + sharpening | Together avoid the degenerate predictions seen with inadequate output balancing |
| Image retrieval and copy detection | Frozen features retain useful instance information |
| Video object segmentation | Feature correspondences support label propagation across frames |
| Final-layer attention | Several heads separate foreground objects without mask supervision |

_Qualitative summary of Sections 4–5._ The results connect the training design to both global recognition and spatial information.

### Attention maps

<p align="center">
  <img src="https://arxiv.org/html/2104.14294v2/attn6.png" alt="Figure 1: DINO class-token attention" width="650">
</p>

<p align="center"><em>Figure 1. DINO class-token attention.</em></p>

Several heads highlight coherent foreground objects.


_Figure 1._ The spatial structure emerges without segmentation supervision.
