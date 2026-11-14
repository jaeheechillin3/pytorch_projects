# [Paper Review] nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation

> **Authors:** Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein\
> **Institutions:** German Cancer Research Center (DKFZ), Heidelberg University, DeepMind, Heidelberg University Hospital\
> **Publication:** Nature Methods 2021\
> **Link:** [https://www.nature.com/articles/s41592-020-01008-z](https://www.nature.com/articles/s41592-020-01008-z)

*Draft — journal metadata/results are distinguished from the accessible [author preprint](https://arxiv.org/abs/1904.08128v2), used for the detailed outline.*

## Abstract

nnU-Net automates dataset-specific segmentation pipelines rather than introducing a complex backbone.

## Introduction

Biomedical datasets vary in spacing, modality, size, and class balance. Reliable pipelines must coordinate these properties with hardware constraints.

## Results

### Automatic Dataset Adaptation

Map a dataset fingerprint into a pipeline fingerprint using fixed defaults, inferred rules, and empirical selection.

### Diverse Structures and Image Properties

Evaluate heterogeneous organs, lesions, and imaging modalities.

### Comparison with Specialized Pipelines

The journal article evaluates 23 public datasets. The accessible preprint reports an earlier evaluation scope; do not interchange their counts.

### Pipeline Configuration versus Architecture

The KiTS analysis examines configuration choices alongside architectural differences.

### Dataset-dependent Configurations

A strong configuration need not transfer unchanged across datasets.

### Robust Design across Tasks

Use multiple tasks to evaluate design decisions.

### Out-of-the-box Availability

The released framework automates configuration and training for new segmentation datasets.

## Discussion

A systematically configured U-Net provides a demanding baseline.

## Methods

### Dataset Fingerprints

Summarize image shapes, voxel spacing, modalities, classes, and intensity statistics.

### Pipeline Fingerprints

Combine the decisions defining preprocessing, architecture, training, and inference.

### Blueprint Parameters

Use U-Net templates, instance normalization, deep supervision, augmentation, and Dice-plus-cross-entropy loss.

### Inferred Parameters

Coordinate resampling, normalization, topology, patch size, and batch size within memory limits.

### Empirical Parameters

Cross-validation selects configurations/ensembles and useful postprocessing.

### Implementation Details

Compare 2D, full-resolution 3D, and cascaded 3D configurations.

## Supplementary notes

Dataset descriptions and generated pipelines document reproducibility.
