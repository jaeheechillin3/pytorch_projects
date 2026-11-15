# [Paper Review] nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation

> **Authors:** Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein\
> **Institutions:** German Cancer Research Center (DKFZ), Heidelberg University, DeepMind, Heidelberg University Hospital\
> **Publication:** Nature Methods 2021\
> **Link:** [https://www.nature.com/articles/s41592-020-01008-z](https://www.nature.com/articles/s41592-020-01008-z)


## Abstract & Introduction

Biomedical segmentation datasets vary in image dimensions, voxel spacing, modality, and target size. A pipeline that works well for one dataset may perform poorly on another, even when both use a U-Net. Architecture is only one part of the problem; preprocessing, training patches, loss functions, and inference settings must work together.

This paper introduces nnU-Net, a framework that configures a segmentation pipeline for a new dataset automatically. It collects established design choices into a set of defaults and adaptation rules, then uses cross-validation to select the final configuration. Its backbone remains a conventional U-Net.

## Methods

### Dataset and pipeline fingerprints

The framework first summarizes the dataset in a _dataset fingerprint_. This includes image shapes, voxel spacings, modalities, class information, and intensity statistics. A _pipeline fingerprint_ records the corresponding choices for preprocessing, network configuration, training, and inference.

The mapping between the two is organized into three groups. Blueprint parameters provide defaults that work across datasets. Inferred parameters adapt the pipeline using dataset properties and memory constraints. Empirical parameters are selected through training and validation.

### Blueprint parameters

The network uses an encoder–decoder structure with skip connections. Convolutions, instance normalization, and leaky ReLU form its basic processing blocks. Instance normalization supports the small batch sizes common in 3D segmentation. Deep supervision adds training signals at multiple decoder resolutions.

Training combines cross entropy with a Dice loss. Cross entropy supervises individual voxel predictions, while Dice measures overlap between a predicted region and its target. For one class, the overlap can be written as

```math
\operatorname{Dice}(p,y)
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

### Diverse datasets

The published Nature Methods article evaluates nnU-Net on **23 public datasets** spanning different modalities and segmentation targets. It performs strongly across these tasks without manual configuration for each dataset.

The examples include organs, tumors, and microscopic structures. Their differences in scale and appearance test the framework's ability to adapt the complete pipeline.

### Pipeline configuration

The KiTS analysis examines why successful kidney and tumor segmentation systems perform well. Differences in pipeline configuration have a substantial effect, and adding architectural components does not consistently improve over a well-configured baseline.

The result helps explain the framework's performance: its strength comes from coordinating the details that are often left to manual experimentation.

### Configuration across tasks

A choice that improves one dataset can hurt another. The framework therefore combines robust defaults with dataset-specific rules and validation. Evaluating decisions across multiple tasks provides a better basis for those defaults than optimizing a single benchmark.

The released framework makes the configuration process available for new segmentation datasets. Detailed rules and generated pipeline examples are also described in the [author preprint](https://arxiv.org/abs/1904.08128v2).
