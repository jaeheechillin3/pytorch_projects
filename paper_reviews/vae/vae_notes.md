# [Paper Review] Auto-Encoding Variational Bayes

> **Authors:** Diederik P. Kingma, Max Welling\
> **Institutions:** University of Amsterdam\
> **Publication:** ICLR 2014\
> **Link:** https://arxiv.org/abs/1312.6114

## Abstract & Introduction

Latent-variable models require integrating over hidden variables, making likelihood evaluation and posterior inference difficult. Auto-Encoding Variational Bayes learns a generative model and an approximate inference network jointly.

## Methods

The decoder defines $p_\theta(x\mid z)$ with prior $p(z)=\mathcal N(0,I)$. The encoder approximates the posterior with $q_\phi(z\mid x)$. Training maximizes the evidence lower bound (ELBO):

$$
\log p_\theta(x)\geq
\mathcal L(x)=
\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z)).
$$

The first term rewards reconstruction; the second regularizes the posterior toward the prior. For a diagonal Gaussian encoder, sampling is reparameterized as

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad
\epsilon\sim\mathcal N(0,I).
$$

This separates randomness from trainable parameters, enabling backpropagation through samples. The Gaussian KL term is analytic:

$$
D_{\mathrm{KL}}=\frac12\sum_j
\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
$$

Generation samples from the prior and then the decoder.

## Results

On MNIST and Frey Faces, AEVB converges faster and reaches better lower bounds than wake-sleep (Figure 2). Low-dimensional experiments also compare marginal likelihood with Monte Carlo EM (Figure 3).

Two-dimensional latent traversals show organized variation in digits and faces (Figure 4), while prior samples demonstrate generation (Figure 5). The ELBO remains a lower bound, and a diagonal Gaussian limits posterior flexibility.
