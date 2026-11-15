# [Paper Review] Fourier Neural Operator for Parametric Partial Differential Equations

> **Authors:** Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar\
> **Institutions:** California Institute of Technology, Purdue University\
> **Publication:** ICLR 2021\
> **Link:** [https://arxiv.org/abs/2010.08895](https://arxiv.org/abs/2010.08895)


## Abstract & Introduction

Many scientific problems require solving the same partial differential equation (PDE) repeatedly for different initial conditions or coefficients. A numerical solver computes each solution separately, which becomes expensive when thousands of evaluations are needed.

This paper proposes the Fourier Neural Operator (FNO), which learns a mapping between input and solution functions from data. Once trained, the same network predicts solutions for new inputs. Fourier-domain parameterization makes the integral layers efficient and allows the learned parameters to be used at different grid resolutions.

## Methods

### Learning operators

An ordinary neural network maps finite-dimensional vectors to vectors. A neural operator instead represents a map between spaces of functions:

```math
G^\dagger:a\mapsto u,\qquad G_\theta(a)\approx G^\dagger(a).
```

For example, $a$ may be a spatially varying permeability field and $u$ the pressure field that solves a Darcy flow equation. Training examples contain sampled values of both functions on a grid. The desired mapping belongs to the underlying functions, even though its implementation uses discrete samples.

### Neural operator architecture

A local lifting network first maps the input into a wider feature representation, $v_0(x)=P(a(x))$. Repeated layers combine a pointwise linear transformation with an integral operator:

```math
v_{t+1}(x)=\sigma\left(
Wv_t(x)+\int_D\kappa_\theta(x,y,a(x),a(y))v_t(y)\,dy
\right).
```

The integral mixes information across the domain. A final pointwise projection $Q$ maps the features back to the output function. Thus the network lifts the input, repeatedly updates a field of features, and projects it to the solution.

### Fourier layers

FNO restricts the integral kernel to a convolution and learns its Fourier representation. The convolution theorem lets the network apply this global operation through multiplication in frequency space:

```math
v_{t+1}
=\sigma\left(Wv_t+
\mathcal F^{-1}\!\left(R_\theta\cdot\mathcal F(v_t)\right)\right).
```

<p align="center">
  <img src="https://arxiv.org/html/2010.08895v3/figs/FourierNN_NV2.png" alt="Fourier layer and Navier–Stokes predictions" width="600">
</p>

<p align="center"><em>Figure 1. Fourier layer architecture and an example of resolution transfer.</em></p>

First, an FFT transforms each feature channel. The layer retains a selected set of low-frequency modes and applies a learned complex-valued matrix to the channels at each retained frequency. An inverse FFT returns the result to the spatial domain, where it is combined with the pointwise branch and passed through a nonlinearity.

The spectral branch communicates globally across the grid. The pointwise branch preserves local information, and spatial nonlinearities can generate higher frequencies across successive layers. Truncating the learned spectral multipliers therefore does not require the final output to contain only the retained frequencies.

For a fixed number of retained modes and feature channels, the FFT gives approximately $O(n\log n)$ complexity in the number of grid points. The same spectral weights can be applied on compatible grids of different resolutions.

## Results

### Burgers' equation

The first experiment learns a map from an initial velocity field to the solution at a later time. Burgers' equation combines nonlinear advection and diffusion. FNO maintains low error across the tested resolutions and outperforms the compared finite-dimensional and operator-learning baselines.

### Darcy flow

The second experiment maps a permeability field to the pressure solving a steady elliptic PDE:

```math
-\nabla\cdot(a(x)\nabla u(x))=f(x).
```

The experiment uses square-domain inputs with specified boundary conditions. It tests whether the network can infer a complete solution field from a spatially varying coefficient. FNO again performs well as the sampling resolution changes.

### Navier–Stokes equation

The time-dependent experiment predicts future vorticity from an observed initial sequence. FNO-2D performs spatial operations and advances recurrently in time; FNO-3D performs operations across space and time to predict a trajectory together.

At viscosity $10^{-3}$ with 1,000 training examples, FNO-3D achieves **0.0086 relative error**, compared with 0.0245 for U-Net. At viscosity $10^{-4}$, increasing the training set from 1,000 to 10,000 examples reduces FNO-3D's error from 0.1918 to 0.0820. More difficult dynamics require more data, and FNO-2D performs better in some of the smaller-data settings.

### Zero-shot super-resolution

A model trained on a $64\times64\times20$ space-time grid is evaluated on a $256\times256\times80$ grid without retraining. The predicted flow remains close to the higher-resolution reference. This demonstrates transfer between discretizations of the studied problem.

### Bayesian inverse problem

The trained operator also replaces repeated numerical forward solves inside a Bayesian inverse problem. Fast forward evaluations make posterior sampling more practical. The saving applies after the cost of producing training solutions and training the network has been paid.
