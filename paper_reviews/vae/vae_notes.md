# [Paper Review] Auto-Encoding Variational Bayes

> **Authors:** Diederik P. Kingma, Max Welling\
> **Institutions:** University of Amsterdam\
> **Publication:** ICLR 2014\
> **Link:** https://arxiv.org/abs/1312.6114


## Abstract & Introduction

Latent-variable models assume that an observation is generated from an unobserved representation. A simple prior can be sampled easily, while a neural network maps each latent sample to a distribution over observations. The difficulty is learning the model: evaluating an observation's likelihood requires integrating over the latent variable, and the exact posterior is generally intractable.

This paper introduces Auto-Encoding Variational Bayes (AEVB), which jointly learns a generative model and an approximate inference model. Its Stochastic Gradient Variational Bayes (SGVB) estimator makes the variational objective differentiable through latent samples. A variational auto-encoder (VAE) is an application of this method using neural networks as the encoder and decoder.

## Methods

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
q_\phi(z\mid x)=\mathcal N\!\left(z\mid\mu_\phi(x),\operatorname{diag}(\sigma_\phi^2(x))\right),
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

### Likelihood lower bound

The model was evaluated on MNIST handwritten digits and Frey Face images. AEVB was compared with wake-sleep across several latent dimensions. AEVB converged faster and reached higher lower bounds in the reported experiments. Increasing the number of latent variables did not lead to the expected overfitting, which the authors attribute to regularization from the variational objective.

### Marginal likelihood

A separate experiment compared estimated marginal likelihoods against wake-sleep and Monte Carlo EM. This evaluation used only three latent dimensions because the likelihood estimator became unreliable in larger latent spaces. The experiment supports the optimization results while also showing the difficulty of evaluating latent-variable models directly.

### Learned latent space

With two latent dimensions, the decoder can be evaluated over a grid to visualize the learned manifold. The MNIST and Frey Face examples show gradual changes in digit appearance and facial expression. These visualizations illustrate how a continuous latent representation organizes observations.

The paper's [Figures 2–3 and Appendix A](https://arxiv.org/pdf/1312.6114) show the optimization comparisons and latent-space visualizations.
