# [Paper Review] Fourier Neural Operator for Parametric Partial Differential Equations

> **Authors:** Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar\
> **Institutions:** California Institute of Technology, Purdue University\
> **Publication:** ICLR 2021\
> **Link:** [https://arxiv.org/abs/2010.08895](https://arxiv.org/abs/2010.08895)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

FNO learns PDE solution operators using Fourier-domain kernels.

## 1 Introduction

One trained surrogate predicts solutions for varying input functions.

## 2 Learning Operators

Learn $G_\theta:a\mapsto u$ from paired input/solution fields. Discretized samples represent functions, rather than independent scalar observations.

## 3 Neural Operator

Lift inputs into features, apply repeated updates, then project to the solution:

```math
v_{t+1}(x)=\sigma\left(
Wv_t(x)+\int_D\kappa_\theta(x,y,a(x),a(y))v_t(y)\,dy
\right).
```

## 4 Fourier Neural Operator

Use a convolutional kernel parameterized by learned Fourier multipliers:

```math
v_{t+1}
=\sigma\left(Wv_t+\mathcal F^{-1}
\left(R_\theta\cdot\mathcal F(v_t)\right)\right).
```

Retain selected modes, mix channels, and transform back. FFT implementation gives approximately $O(n\log n)$ spatial complexity.

Shared spectral weights allow resolution transfer; accuracy still depends on sampling and distribution.

## 5 Numerical Experiments

### 5.1 Burgers' Equation

Map initial conditions to later solutions.

### 5.2 Darcy Flow

Map permeability fields to pressure solutions.

### 5.3 Navier–Stokes Equation

Predict vorticity evolution; compare space-only and space–time variants.

### 5.4 Zero-shot Super-resolution

Evaluate finer grids without retraining.

### 5.5 Bayesian Inverse Problem

Use the surrogate for repeated forward evaluations during inference.

## 6 Discussion and Conclusion

Operator learning amortizes repeated PDE solves. Report prediction speed separately from training/data-generation costs.

## Appendix notes

Consult data-generation, spectral-analysis, boundary-treatment, and inverse-problem details.
