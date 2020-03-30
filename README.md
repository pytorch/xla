# PyTorch/XLA

<b>Current CI status:</b>  [![CircleCI](https://circleci.com/gh/pytorch/xla.svg?style=svg)](https://circleci.com/gh/pytorch/xla)

PyTorch/XLA is a Python package that uses the
[XLA deep learning compiler](https://www.tensorflow.org/xla)
to connect the [PyTorch deep learning framework](https://pytorch.org/) and
[Cloud TPUs](https://cloud.google.com/tpu/). You can try it right now, for free,
on a single Cloud TPU with [Google Colab](https://colab.research.google.com/),
and use it in production and on Cloud TPU Pods
with [Google Cloud](https://cloud.google.com/gcp).

Take a look at one of our Colab notebooks to quickly try different PyTorch networks
running on Cloud TPUs and learn how to use Cloud TPUs as PyTorch devices:

* [Getting Started with PyTorch on Cloud TPUs](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb)
* [Training AlexNet on Fashion MNIST with a single Cloud TPU Core](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/single-core-alexnet-fashion-mnist.ipynb)
* [Training AlexNet on Fashion MNIST with multiple Cloud TPU Cores](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/multi-core-alexnet-fashion-mnist.ipynb)
* [Fast Neural Style Transfer (NeurIPS 2019 Demo)](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/style_transfer_inference.ipynb)
* [Training A Simple Convolutional Network on MNIST](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/mnist-training.ipynb)
* [Training a ResNet18 Network on CIFAR10](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/resnet18-training.ipynb)
* [ImageNet Inference with ResNet50](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/resnet50-inference.ipynb)

The rest of this README covers:

* [Running PyTorch on Cloud TPUs in production on Google Cloud.](#Cloud)
Google Cloud also runs networks faster than Google Colab.
* [API & Best Practices](#API)
* [Troubleshooting](#Troubleshooting)
* [Providing Feedback](#Feedback)
* [Building and Contributing to PyTorch/XLA](#Contributing)

Additional information on PyTorch/XLA, including a description of its
semantics and functions, is available at [PyTorch.org](http://pytorch.org/xla/).

## <a name="Cloud"></a> Running PyTorch on Cloud TPUs with Google Cloud Platform

Google Cloud Platform lets you deploy PyTorch networks running on Cloud TPUs.
This guide is split into two parts:

* [Runnining on a single Cloud TPU](#CloudSingle)
* [Running on a Cloud TPU Pod](#Pod)

## <a name="CloudSingle"></a> Running on a Single Cloud TPU

The following tutorials are available to help you train models on a single
Cloud TPU:

* [Training FairSeq Transformer on Cloud TPUs](https://cloud.google.com/tpu/docs/tutorials/transformer-pytorch)
* [Training Resnet50 on Cloud TPUs](https://cloud.google.com/tpu/docs/tutorials/resnet-alpha-py)

To start, [you create a Cloud TPU node](https://cloud.google.com/tpu/docs/tutorials/resnet-alpha-py#create_tpu) with the corresponding release you wish to consume (TPU software version: ex. `pytorch-1.5`):

Once you've created a Cloud TPU node, you can train your PyTorch models by either:

* [Consuming prebuilt docker images (*recommended*)](#DockerImage)
* [Consuming prebuilt Compute VM Images](#VMImage)


### <a name="DockerImage"></a> Consume Prebuilt Docker Images

Follow these steps to train a PyTorch model with Docker on a Cloud TPU:

1. Create a Compute VM and install docker (or use COS VM image)
    * *Note: make sure the Compute VM is within the **same** zone as the TPU node you created or else performance will suffer, also ideally create a VM that has at least 16 cores (`n1-standard-16`) to not be VM compute/network bound.*

    Docker images with `torch` and `torch_xla` preinstalled in the `pytorch` conda
    environment are distributed under: `gcr.io/tpu-pytorch/xla`.

2. SSH into the VM and pull a version of the docker image into the VM. The currently available versions are:

    * `gcr.io/tpu-pytorch/xla:r1.5`: The current stable version.
    * `gcr.io/tpu-pytorch/xla:nightly`: Nightly version.
    * `gcr.io/tpu-pytorch/xla:nightly_YYYYMMDD (e.g.: gcr.io/tpu-pytorch/xla:nightly_20190531)`: The nightly version of the given day.

    At this time is recommended to use nightly versions and eventually switch to the stable version in case there are issues with nightly.
    Remember to create a TPU with `pytorch-nightly` version when using nightly.

    To pull the dockers run one of the following commands:

    ```Shell
    (vm)$ docker pull gcr.io/tpu-pytorch/xla:nightly
    ```

    ```Shell
    (vm)$ docker pull gcr.io/tpu-pytorch/xla:nightly_YYYYMMDD
    ```

    ```Shell
    (vm)$ docker pull gcr.io/tpu-pytorch/xla:r1.5
    ```

3. Where `$TPU_IP_ADDRESS` (e.g.: `10.1.1.2`) is your TPU Internal IP displayed in GCP UI, after pulling the docker image you can either:

    * Run the container with a single command:
      ```Shell
      (vm)$ docker run --shm-size 16G -e XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470" gcr.io/tpu-pytorch/xla:r1.5 python /pytorch/xla/test/test_train_mnist.py
      ```

    * Run the script in an interactive shell:
      ```Shell
      (vm)$ docker run -it --shm-size 16G gcr.io/tpu-pytorch/xla:r1.5
      (pytorch) root@CONTAINERID:/$ export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
      (pytorch) root@CONTAINERID:/$ python pytorch/xla/test/test_train_mnist.py
      ```

### <a name="VMImage"></a> Consume Prebuilt Compute VM Images

Follow these steps to train a PyTorch model with a VM Image on a Cloud TPU:

1. Create a Compute VM with PyTorch/XLA Image.

    * In the GCP Console, go to the [**VM Instances**](https://console.cloud.google.com/compute/instances) page.
    * Click **Create Instance**.
    * Make sure the compute VM is within the **same** zone as the TPU node you created or else performance will suffer, also ideally create a VM that has at least 16 cores (`n1-standard-16`) to not be VM compute/network bound.
    * In the **Boot disk** section, click **Change** to choose our PyTorch/XLA image.
    * At the bottom of the **OS Images** tab select the **Debian GNU/Linux 9 Stretch + PyTorch/XLA** image.
    * Chose an appropriate dist size based on your dataset and click **Select**.
    * Click **Create** to create the instance.


2. SSH into VM and activate the conda environment you wish to use. Each release (e.g.: `0.1`, `0.5`, `1.5`, `nightly`) is a separate conda environment.

    ```Shell
    (vm)$ export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    (vm)$ conda env list
    # conda environments:
    #
    base                  *  /anaconda3
    torch-xla-0.1              /anaconda3/envs/torch-xla-0.1
    torch-xla-0.5              /anaconda3/envs/torch-xla-0.5
    torch-xla-1.5              /anaconda3/envs/torch-xla-1.5
    torch-xla-nightly          /anaconda3/envs/torch-xla-nightly

    (vm)$ conda activate torch-xla-1.5
    (torch-xla-1.5)$ cd /usr/share/torch-xla-1.5/pytorch/xla
    (torch-xla-1.5)$ python test/test_train_mnist.py
    ```

    To update the wheels `torch` and `torch_xla` to the latest nightly
    distribution (only updates your `torch-xla-nightly` conda env), run:
    ```Shell
    (vm)$ cd /usr/share/torch-xla-nightly/pytorch/xla
    (vm)$ . ./scripts/update_nightly_torch_wheels.sh
    ```

---

## <a name="Pod"></a> How to Run on TPU Pods (distributed training)

Whereas the previous section focused on training on a single TPU node,
this section discusses distributed training in TPU Pods. The tutorial,
[Training PyTorch models on Cloud TPU Pods](https://cloud.google.com/tpu/docs/tutorials/pytorch-pod), is a great place to start.

The recommended setup for running distributed training on TPU Pods uses the
pairing of Compute VM [Instance
Groups](https://cloud.google.com/compute/docs/instance-groups/) and TPU Pods.
Each of the Compute VM in the instance group drives 8 cores on the TPU Pod and
so using an instance group ensures each of the Compute VMs use the identical
base image.

Training on pods can be broken down to largely 3 different steps:
1. [Create your instance group (*recommended*)](#create-your-instance-group) or [Use a list of
   VM instances](#list-of-vms)
2. [Create your TPU Pod](#create-your-tpu-pod)
3. [Start distributed training](#start-distributed-training)

### Create your instance group

1. Create an instance template.
* During creation, make sure to go to section "Identity and API access" → "Access Scopes" and select "Allow full access to all Cloud APIs".
* If you have already have a VM instance running that you used to train PyTorch/TPU workloads and want to use that exact setup for distributed training: [instructions](https://cloud.google.com/compute/docs/instance-templates/create-instance-templates#based-on-existing-instance).
* Or, you can create an instance template using the PyTorch/XLA VM image we provide: [instructions](https://cloud.google.com/compute/docs/instance-templates/create-instance-templates#creating_a_new_instance_template).
2. Create an instance group to drive the TPU pod.
* This instance group is where all the input pipeline happens and where we feed all the tensors into the TPUs for training.
* Use the instance template created in step (1) to create your instance group.
* Make sure to (a) create the instance group in a single zone (same zone as the TPU Pod you'll create), (b) no autoscaling or health-checks, (c) number of instances (size of instance group) should be number of cores / 8 (ex. for a v3-32 you'd create an instance group of size 32/8 = 4).
* Here are the instructions for creating an instance group: [instructions](https://cloud.google.com/compute/docs/instance-groups/creating-groups-of-managed-instances#create_managed_group).

### Create your TPU Pod
1. [Create](https://pantheon.corp.google.com/compute/tpus) a TPU pod (same as creating regular TPUs, just select more cores when selecting TPU type).
* Make sure that the TPU is in the same zone as the instance group.
* Make sure that the size of your instance group follows: # instances in group = number of TPU cores / 8.

### Start distributed training
1. SSH into any of the VMs in the instance group and get in an environment where you have `torch` and `torch_xla` installed (whether that's a [conda environment](#consume-prebuilt-compute-vm-images) or [docker container](#consume-prebuilt-docker-images)).
2. Let's say the command you ran to run a v3-8 was: `XLA_USE_BF16=1 python test/test_train_imagenet.py --fake_data`.
* To distribute training as a conda environment process:
```
(torch-xla-nightly)$ python -m torch_xla.distributed.xla_dist --tpu=$TPU_POD_NAME --conda-env=torch-xla-nightly --env=XLA_USE_BF16=1 -- python /usr/share/torch-xla-0.5/pytorch/xla/test/test_train_imagenet.py --fake_data
```

* Or, to distribute training as a docker container:
```
(torch-xla-nightly)$ cd /usr/share/torch-xla-nightly/pytorch/xla
(torch-xla-nightly)$ python -m torch_xla.distributed.xla_dist --tpu=$TPU_POD_NAME --docker-image=gcr.io/tpu-pytorch/xla:nightly --docker-run-flag=--rm=true --docker-run-flag=--shm-size=50GB --env=XLA_USE_BF16=1 -- python test/test_train_imagenet.py --fake_data
```

### List of VMs
If you up to not use an [instance group](#create-your-instance-group), you can decide to use a list of VM instances that you may have already created (or can create individually). Make sure that you create all the VM instances in the same zone as the TPU node, and also make sure that the VMs have the same configuration (datasets, VM size, disk size, etc.). Then you can [start distributed training](#start-distributed-training) after creating your TPU pod. The difference is in the `python -m torch_xla.distributed.xla_dist` command. For example, to use a list of VMs run the following command (ex. conda with v3-32):
```
(torch-xla-nightly)$ cd /usr/share/torch-xla-nightly/pytorch/xla
(torch-xla-nightly)$ python -m torch_xla.distributed.xla_dist --tpu=$TPU_POD_NAME --vm $VM1 --vm $VM2 --vm $VM3 --vm $VM4 --conda-env=torch-xla-nightly --env=XLA_USE_BF16=1 -- python test/test_train_imagenet.py --fake_data
```

To learn more about TPU Pods check out this [blog
post](https://cloud.google.com/blog/products/ai-machine-learning/googles-scalable-supercomputers-for-machine-learning-cloud-tpu-pods-are-now-publicly-available-in-beta). For more information regarding system architecture, please refer to the
[Cloud TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture) page.

## <a name="API"></a> API & Best Practices

See the [API Guide](API_GUIDE.md) for best practices when writing networks that
run on Cloud TPUs and Cloud TPU Pods.

## <a name="Troubleshooting"></a> Troubleshooting

If PyTorch/XLA isn't performing as expected, see the
[troubleshooting guide](TROUBLESHOOTING.md), which has suggestions for
debugging and optimizing your network(s).

## <a name="Feedback"></a> Providing Feedback

The PyTorch/XLA team is always happy to hear from users and OSS contributors!
The best way to reach out is by filing an issue on this Github. Questions,
bug reports, feature requests, build issues, etc. are all welcome!

## <a name="Contributing"></a> Contributing

See the [contribution guide](CONTRIBUTING.md).
