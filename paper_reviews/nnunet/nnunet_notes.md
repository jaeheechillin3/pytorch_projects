# [Paper Review] nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation

> **Authors:** Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein\
> **Institutions:** German Cancer Research Center (DKFZ), Heidelberg University, DeepMind, Heidelberg University Hospital\
> **Publication:** Nature Methods 2021\
> **Link:** [https://www.nature.com/articles/s41592-020-01008-z](https://www.nature.com/articles/s41592-020-01008-z)

## Abstract & Introduction

A segmentation model is only as good as the pipeline around it. nnU-Net turns the many interdependent choices in biomedical segmentation—resampling, normalization, network shape, patches, training, and postprocessing—into defaults, adaptation rules, and validation decisions. A conventional U-Net configured this way performs strongly across diverse datasets, making systematic configuration a demanding baseline for new architectures.

## Methods

<p align="center">
  <img src="https://media.springernature.com/lw1200/springer-static/image/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_Fig2_HTML.png" alt="Figure 2. nnU-Net's automated pipeline design and configuration." width="700">
</p>

<p align="center"><em>Figure 2. nnU-Net's automated pipeline design and configuration. From the <a href="https://www.nature.com/articles/s41592-020-01008-z">original paper</a>.</em></p>

### Dataset and pipeline fingerprints

The framework first summarizes the dataset in a _dataset fingerprint_. This includes image shapes, voxel spacings, modalities, class information, and intensity statistics. A _pipeline fingerprint_ records the corresponding choices for preprocessing, network configuration, training, and inference.

The mapping between the two is organized into three groups. Blueprint parameters provide defaults that work across datasets. Inferred parameters adapt the pipeline using dataset properties and memory constraints. Empirical parameters are selected through training and validation.

### Blueprint parameters

The network uses an encoder–decoder structure with skip connections. Convolutions, instance normalization, and leaky ReLU form its basic processing blocks. Instance normalization supports the small batch sizes common in 3D segmentation. Deep supervision adds training signals at multiple decoder resolutions.

Training combines cross entropy with a Dice loss. Cross entropy supervises individual voxel predictions, while Dice measures overlap between a predicted region and its target. For one class, the overlap can be written as

```math
\mathrm{Dice}(p,y)
=\frac{2\sum_i p_i y_i+\epsilon}
{\sum_i p_i+\sum_i y_i+\epsilon}.
```

Here $p_i$ is a soft prediction and $y_i$ is the target indicator. The expression illustrates the overlap term; class aggregation and smoothing follow the implementation.

Spatial and intensity augmentation improve robustness. Foreground oversampling makes training patches more likely to contain the structures being segmented, which is useful when an organ or lesion occupies only a small part of the image.

### Inferred parameters

Voxel spacing determines how much physical space each voxel represents. The framework chooses a target spacing and resamples images and labels accordingly. Strongly anisotropic datasets require different treatment along their low-resolution axis; early operations can remain effectively two-dimensional until the feature spacing becomes more balanced.

Intensity normalization also depends on modality. CT intensities have a consistent physical interpretation, allowing dataset-level statistics, while MRI intensities are generally normalized per image.

Patch size, network depth, pooling, and batch size are chosen together within the available memory. Larger patches provide more anatomical context but consume more memory. The configuration rules balance this context against the need to train a sufficiently deep network.

### Candidate configurations and selection

nnU-Net considers a 2D U-Net, a full-resolution 3D U-Net, and, where appropriate, a 3D cascade. The cascade first segments a lower-resolution volume. A second network receives the full-resolution image together with the coarse prediction, adding global context when full-resolution patches cover too little anatomy.

Cross-validation determines which configuration or ensemble performs best. Validation also determines whether postprocessing, such as retaining the largest connected component for a class, improves results. This prevents a rule that helps one dataset from being applied indiscriminately to every task.

Inference combines predictions from overlapping patches. Weighting reduces artifacts near patch boundaries, and test-time mirroring can average predictions across transformed inputs.

## Results

### Breadth of evaluation

| Evaluation | Scope / finding |
| --- | --- |
| Published journal study | 23 public datasets |
| Imaging domains | Includes CT, MRI, and microscopy |
| Target structures | Includes organs, tumors, lesions, and cellular structures |
| Configuration | Automated preprocessing, architecture, training, and postprocessing |
| Overall comparison | Surpasses most compared approaches, including specialized pipelines |

_Summary of the journal article's evaluation._ The same configuration framework works across substantially different image properties and segmentation targets.

### What drives performance?

| Analysis | Reported finding |
| --- | --- |
| KiTS pipeline comparison | Configuration details have a substantial influence on performance |
| Architectural extensions | Added complexity does not consistently beat a well-configured U-Net |
| Cross-dataset choices | A configuration that helps one dataset can hurt another |
| Multi-task design | Testing across tasks provides more reliable defaults |
| Validation-based selection | Chooses useful configurations, ensembles, and postprocessing |

_Qualitative summary of the Results section._ The evidence supports coordinated pipeline design as the central contribution, rather than a particular new network block.
