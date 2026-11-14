# [Paper Review] Auto-Encoding Variational Bayes

> **Authors:** Diederik P. Kingma, Max Welling\
> **Institutions:** University of Amsterdam\
> **Publication:** ICLR 2014\
> **Link:** [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

*Draft — follows the paper's main section order; appendices are condensed.*

## Abstract

AEVB combines approximate inference with differentiable stochastic optimization.

## 1 Introduction

Latent-variable likelihoods require integrating over hidden variables. A learned encoder supplies approximate posteriors.

## 2 Method

### 2.1 Problem Scenario

```math
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.
```

Both marginal likelihood and exact posterior may be intractable.

### 2.2 The Variational Bound

```math
\log p_\theta(x)
=\mathcal L(\theta,\phi;x)
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right),
```

```math
\mathcal L
=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right).
```

Nonnegative KL makes $\mathcal L$ a lower bound.

### 2.3 The SGVB Estimator and AEVB Algorithm

Estimate expectations with samples and jointly update encoder/decoder parameters using minibatches.

### 2.4 The Reparameterization Trick

Move randomness into parameter-independent noise:

```math
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad
\epsilon\sim\mathcal N(0,I).
```

Gradients then pass through the deterministic transformation.

## 3 Example: Variational Auto-Encoder

Use a standard-normal prior, diagonal-Gaussian encoder, and Bernoulli or Gaussian decoder.

```math
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|\mathcal N(0,I)\right)
=\frac12\sum_j\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
```

Minimize negative ELBO; the reconstruction loss follows the decoder likelihood.

## 4 Related Work

Compare with wake–sleep, variational inference, and stochastic gradient estimators.

## 5 Experiments

**Likelihood lower bound.** Compare optimization on MNIST and Frey Face.

**Marginal likelihood.** Distinguish estimated likelihood from its bound.

**Visualization.** Inspect low-dimensional latent manifolds.

## 6 Conclusion

Learn inference and generation together.

## 7 Future Work

Extend the framework to richer probabilistic models.

## Appendix notes

Gaussian KL derivation, encoder/decoder parameterizations, and likelihood estimation support the main method.
