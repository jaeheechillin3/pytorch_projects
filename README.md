# PyTorch Learning Projects

This repository is a learning log of my progress with PyTorch. Each notebook focuses on a small set of concepts, moving from tensor fundamentals and automatic differentiation to image classification and generative models.

## Progress

| Status | Project | Notebook | Main focus |
|---|---|---|---|
| Completed | PyTorch fundamentals | [`vid1.ipynb`](vid1.ipynb), [`vid2.ipynb`](vid2.ipynb), [`vid3.ipynb`](vid3.ipynb) | Tensors, neural-network basics, data loading, and autograd |
| Completed | Linear regression | [`linear_regression.ipynb`](linear_regression.ipynb) | Building and optimizing a model manually |
| Completed | MNIST logistic regression | [`mlp_mnist.ipynb`](mlp_mnist.ipynb) | Image classification with a linear classifier |
| Completed | LeNet | [`lenet.ipynb`](lenet.ipynb) | Convolutional neural networks for MNIST |
| Completed | Autoencoder | [`autoencoder.ipynb`](autoencoder.ipynb) | Image reconstruction and latent representations |
| Planned | U-Net | — | Semantic segmentation |
| Planned | Variational autoencoder | — | Probabilistic latent-variable models |

## What I learned

### 1. PyTorch fundamentals

These notebooks follow the [Introduction to PyTorch](https://www.youtube.com/watch?v=IC0_FRiX-sw&list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN) video series.

- Creating tensors with different shapes and data types
- Using tensor factory functions such as `zeros`, `ones`, `rand`, `empty`, and their `_like` variants
- Setting random seeds for reproducible results
- Performing element-wise operations and basic linear-algebra operations
- Defining models by subclassing `torch.nn.Module` and implementing `forward()`
- Combining convolution, activation, pooling, and fully connected layers
- Loading and batching CIFAR-10 data with `Dataset` and `DataLoader`
- Understanding autograd, computation graphs, `requires_grad`, and `backward()`

### 2. Linear regression

Based on [Training a Linear Regression Model in PyTorch](https://machinelearningmastery.com/training-a-linear-regression-model-in-pytorch/).

- Generating synthetic linear data and adding Gaussian noise
- Writing a forward function for the model equation
- Measuring error with mean squared error
- Making a parameter learnable with `requires_grad=True`
- Computing gradients through backpropagation
- Updating a parameter manually with gradient descent
- Clearing accumulated gradients between iterations
- Tracking and plotting loss during training

The model starts with a slope of `-10` and learns a value close to the target slope of `-5` as the loss decreases.

### 3. MNIST logistic regression

Based on [Building a Logistic Regression Classifier in PyTorch](https://machinelearningmastery.com/building-a-logistic-regression-classifier-in-pytorch/).

- Loading and inspecting the MNIST dataset through `torchvision`
- Converting images to tensors and understanding the `[1, 28, 28]` image shape
- Using `DataLoader` for batching and shuffling
- Flattening each image from `28 × 28` pixels to a 784-element vector
- Creating a custom classifier with `nn.Module` and `nn.Linear`
- Training with stochastic gradient descent and cross-entropy loss
- Evaluating predictions on the test set
- Recording and plotting loss and accuracy across epochs

The notebook records a final test accuracy of **86.73%** after 50 epochs. Despite the filename `mlp_mnist.ipynb`, the current model is a single-layer logistic regression classifier rather than a multilayer perceptron.

### 4. LeNet

Based on the LeNet tutorial in [bentrevett/pytorch-image-classification](https://github.com/bentrevett/pytorch-image-classification).

- Calculating dataset statistics and normalizing MNIST images
- Applying data augmentation with random rotations and crops
- Splitting data into training, validation, and test sets
- Exploring convolution filters for horizontal, vertical, and diagonal edges
- Comparing max pooling and average pooling
- Following tensor dimensions through convolution and pooling layers
- Implementing the LeNet architecture with convolutional and fully connected layers
- Writing separate training and evaluation loops
- Training with Adam and cross-entropy loss
- Saving the weights with the best validation loss
- Inspecting learned convolution filters

The trained model contains **44,426 trainable parameters** and records **99.29% test accuracy** on MNIST.

### 5. Autoencoder

- Building an encoder-decoder network for MNIST reconstruction
- Compressing images into latent representations
- Comparing bottleneck sizes of 2, 8, and 24 dimensions
- Using ReLU in hidden layers and sigmoid for reconstructed pixel values
- Training with Adam and mean squared reconstruction error
- Selecting CUDA, Apple MPS, or CPU automatically
- Making experiments reproducible with fixed random seeds
- Comparing training curves and reconstructed images across latent dimensions

After 10 epochs, the recorded training MSE values were **0.041732**, **0.021151**, and **0.014443** for latent dimensions 2, 8, and 24, respectively. The larger latent space preserved more information and produced a lower reconstruction error.

## Planned projects

### 6. U-Net

The planned U-Net project will focus on encoder-decoder architectures, skip connections, pixel-wise prediction, semantic segmentation, and evaluation with metrics such as Dice score and intersection over union.

### 7. Variational autoencoder

The planned VAE project will focus on probabilistic latent spaces, learning mean and variance parameters, the reparameterization trick, combining reconstruction and KL-divergence losses, and generating new samples.

## Running the notebooks

The notebooks are designed to run in Google Colab or a local Jupyter environment. The datasets used by the notebooks are downloaded automatically through `torchvision`.

Main dependencies:

```text
torch
torchvision
numpy
matplotlib
scikit-learn
tqdm
```
