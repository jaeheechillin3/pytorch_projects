# [Paper Review] Auto-Encoding Variational Bayes

> **Authors:** Diederik P. Kingma, Max Welling\
> **Institutions:** University of Amsterdam\
> **Publication:** ICLR 2014\
> **Link:** https://arxiv.org/abs/1312.6114

## Abstract & Introduction

Learning a latent-variable model becomes difficult when its posterior and marginal likelihood cannot be evaluated directly. Auto-Encoding Variational Bayes jointly trains a generative model and a probabilistic encoder by optimizing a variational lower bound. Reparameterizing latent samples makes the objective differentiable through both networks, yielding an efficient method for inference, representation learning, and generation.

## Methods

<p align="center">
  <img src="https://hunterheidenreich.com/img/notes/autoencoding-variational-bayes-figure-1-model-diagram.webp" alt="Figure 1. Generative model and variational approximation." width="700">
</p>

<p align="center"><em>Figure 1. Generative model and variational approximation. From the <a href="https://arxiv.org/abs/1312.6114">original paper</a>.</em></p>

### Generative model and probabilistic encoder

The model first samples a latent vector from a prior, then samples an observation from the decoder:

```math
p_\theta(x,z)=p_\theta(z)p_\theta(x\mid z),\qquad
p_\theta(x)=\int p_\theta(x,z)\,dz.
```

The encoder $q_\phi(z\mid x)$ approximates the model posterior $p_\theta(z\mid x)$. Its parameters are shared across observations, so inference for a new example requires a network evaluation. The encoder produces the parameters of a distribution over representations.

### The variational lower bound

The log likelihood decomposes into a lower bound and a nonnegative posterior approximation error:

```math
\log p_\theta(x)
=\mathcal L(\theta,\phi;x)
+D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right).
```

```math
\mathcal L(\theta,\phi;x)
=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right).
```

The first term rewards explaining the observation from its latent representation. The second penalizes encoder distributions that depart from the prior. Maximizing the bound trains both networks; for a fixed generative model, it also brings the encoder closer to the exact posterior. Training minimizes the negative of this bound.

### Reparameterization and stochastic optimization

Directly sampling from a distribution whose parameters depend on the encoder makes ordinary backpropagation difficult. The paper expresses the sample as a differentiable function of the parameters and independent noise. For a diagonal Gaussian encoder,

```math
q_\phi(z\mid x)=\mathcal N\!\left(z\mid\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x))\right),
\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\quad
\epsilon\sim\mathcal N(0,I).
```

The noise distribution is fixed. Gradients can therefore pass through the sampled representation to the encoder's mean and standard deviation. Averaging the reconstruction term over sampled noise gives a differentiable Monte Carlo estimate.

For the standard normal prior, the KL term is available analytically:

```math
D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))
=\frac12\sum_j\left(\mu_{\phi,j}^2+\sigma_{\phi,j}^2-1-\log\sigma_{\phi,j}^2\right).
```

Only the expected decoder log likelihood needs sampling. The paper finds that one latent sample per observation can suffice when the minibatch is large enough. Minibatches make the method practical for large datasets.

### VAE implementation

The experiments use multilayer perceptrons for both networks. The decoder uses a Bernoulli likelihood for binary data and a Gaussian likelihood for continuous data. These choices determine the reconstruction loss: binary cross entropy for the Bernoulli model and a variance-weighted squared error with a log variance term for the Gaussian model.

Generation uses the learned decoder after drawing a fresh latent vector from the prior. Reconstruction instead begins with an observation and draws its latent representation from the encoder.

## Results

### Inference and learning

| Experiment | Setting | Reported outcome |
| --- | --- | --- |
| Lower-bound optimization | MNIST and Frey Face; several latent dimensions | AEVB converges faster and reaches better bounds than wake-sleep |
| Latent capacity | Increasing the number of latent variables | Extra latent dimensions do not produce the expected overfitting in the reported runs |
| Marginal-likelihood comparison | MNIST with 3 latent dimensions | AEVB compares favorably with wake-sleep and Monte Carlo EM |
| Likelihood-estimator reliability | Increasing latent dimension | Estimates become unreliable, restricting this comparison to low dimension |

_Qualitative summary of Figures 2–3 and Section 5._ The experiments support efficient optimization, while the evaluation itself becomes harder in larger latent spaces.

### Learned manifolds

| Dataset | Visualization | Observation |
| --- | --- | --- |
| MNIST | <img src="https://arxiv.org/html/1312.6114v11/mnist_20x20.png" alt="MNIST manifold from a two-dimensional VAE" width="240"> | Neighboring latent coordinates decode to gradual changes in digit appearance |
| Frey Face | <img src="https://arxiv.org/html/1312.6114v11/freyface_10x10.png" alt="Frey Face manifold from a two-dimensional VAE" width="240"> | Nearby codes vary smoothly in expression and appearance |

_Figure 4, shown as separate panels._ The two-dimensional examples make the continuity of the learned generative mapping visible.
