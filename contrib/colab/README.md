# Colab Notebooks

The following are a couple of sample colab notebooks.

## Get started with our Colab Tutorials
* [Getting Started with PyTorch on Cloud TPUs](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb)
* [Training AlexNet on Fashion MNIST with a single Cloud TPU Core](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/single-core-alexnet-fashion-mnist.ipynb)
* [Training AlexNet on Fashion MNIST with multiple Cloud TPU Cores](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/multi-core-alexnet-fashion-mnist.ipynb)
* [Fast Neural Style Transfer (NeurIPS 2019 Demo)](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/style_transfer_inference.ipynb)
* [Training A Simple Convolutional Network on MNIST](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/mnist-training.ipynb)
* [Training a ResNet18 Network on CIFAR10](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/resnet18-training.ipynb)
* [ImageNet Inference with ResNet50](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/resnet50-inference.ipynb)

*Note*: These colab notebooks typically run on small machines (the Compute VMs,
which runs the input pipeline) and training is often bottlenecked on the small
Compute VM machines. For optimal performance create a GCP VM and TPU pair
following our GCP Tutorials:
* [Training FairSeq Transformer on Cloud TPUs](https://cloud.google.com/tpu/docs/tutorials/transformer-pytorch)
* [Training Resnet50 on Cloud TPUs](https://cloud.google.com/tpu/docs/tutorials/resnet-pytorch)
* [Training PyTorch models on Cloud TPU Pods](https://cloud.google.com/tpu/docs/tutorials/pytorch-pod)
