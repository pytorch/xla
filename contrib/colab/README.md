# Colab Notebooks

We have a couple of Colab notebooks here that work with Colab. The name of the
notebooks tells which version of Colab TF/XRT they're compatible with. For
example, `mnist-training-xrt-1-15.ipynb` has been tested to be compatible with
`TF/XRT 1.15`. On colab, you can check the current `TF/XRT` version by running
a snippet like:

```
import tensorflow as tf
tf.__version__
```

*Note*: These colab notebooks typically run on small machines (the Compute VMs,
which runs the input pipeline) and training is often bottlenecked on the small
Compute VM machines. For optimal performance create a GCP VM and TPU pair
following our GCP Tutorials.

## Get started with our Colab Tutorials
* [Training MNIST on TPUs](https://colab.research.google.com/github/pytorch/xla/blob/xrt.r1.15/contrib/colab/mnist-training-xrt-1-15.ipynb)
* [Training ResNet18 on TPUs with Cifar10 dataset](https://colab.research.google.com/github/pytorch/xla/blob/xrt.r1.15/contrib/colab/resnet18-training-xrt-1-15.ipynb)
* [Inference with Pretrained ResNet50 Model](https://colab.research.google.com/github/pytorch/xla/blob/xrt.r1.15/contrib/colab/resnet50-inference-xrt-1-15.ipynb)
