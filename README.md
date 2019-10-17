[![CircleCI](https://circleci.com/gh/pytorch/xla.svg?style=svg)](https://circleci.com/gh/pytorch/xla)

For information regarding system architecture, please refer to the
[Cloud TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture) page.

# How to Run PyTorch with Single TPU Nodes

You can either follow these tutorials available on Google Cloud website:

* [Training FairSeq Transformer on Cloud TPUs](https://cloud.google.com/tpu/docs/tutorials/transformer-pytorch)
* [Training Resnet50 on Cloud TPUs](https://cloud.google.com/tpu/docs/tutorials/resnet-alpha-py)
* [Training PyTorch models on Cloud TPU Pods](https://cloud.google.com/tpu/docs/tutorials/pytorch-pod)

Or the following README to run your model.

First [create your Cloud TPU node](https://cloud.google.com/tpu/docs/tutorials/resnet-alpha-py#create_tpu) with the corresponding release you wish to consume (TPU software version: ex. `pytorch-0.5`):

Once you've created a Cloud TPU node, you can train your PyTorch models by either:

* [Consuming prebuilt docker images (*recommended*)](#consume-prebuilt-docker-images)
* [Consuming prebuilt Compute VM Images](#consume-prebuilt-compute-vm-images)


## Consume Prebuilt Docker Images

Follow these steps to train a PyTorch model with Docker on a TPU:

1. Create a Compute VM and install docker (or use COS VM image)
    * *Note: make sure the Compute VM is within the **same** zone as the TPU node you created or else performance will suffer, also ideally create a VM that has at least 16 cores (`n1-standard-16`) to not be VM compute/network bound.*

    Docker images with `torch` and `torch_xla` preinstalled in the `pytorch` conda
    environment are distributed under: `gcr.io/tpu-pytorch/xla`.

2. SSH into the VM and pull the stable docker image into the VM:

    ```Shell
    (vm)$ docker pull gcr.io/tpu-pytorch/xla:r0.5
    ```

    Note we do also expose the following nightly Docker image versions, but we recommend you use a stable version (`r0.5`):
    * `gcr.io/tpu-pytorch/xla:nightly`
    * `gcr.io/tpu-pytorch/xla:nightly_YYYYMMDD (e.g.: gcr.io/tpu-pytorch/xla:nightly_20190531)`

    If you decide to consume this, be sure to create a TPU with `pytorch-nightly` version.

3. Where `$TPU_IP_ADDRESS` (e.g.: `10.1.1.2`) is your TPU Internal IP displayed in GCP UI, after pulling the docker image you can either:

    * Run the container with a single command:
      ```Shell
      (vm)$ docker run --shm-size 16G -e XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470" gcr.io/tpu-pytorch/xla:r0.5 python /pytorch/xla/test/test_train_mnist.py
      ```

    * Run the script in an interactive shell:
      ```Shell
      (vm)$ docker run -it --shm-size 16G gcr.io/tpu-pytorch/xla:r0.5
      (pytorch) root@CONTAINERID:/$ export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
      (pytorch) root@CONTAINERID:/$ python pytorch/xla/test/test_train_mnist.py
      ```

## Consume Prebuilt Compute VM Images

1. Create a Compute VM with PyTorch/XLA Image.

    * In the GCP Console, go to the [**VM Instances**](https://console.cloud.google.com/compute/instances) page.
    * Click **Create Instance**.
    * Make sure the compute VM is within the **same** zone as the TPU node you created or else performance will suffer, also ideally create a VM that has at least 16 cores (`n1-standard-16`) to not be VM compute/network bound.
    * In the **Boot disk** section, click **Change** to choose our PyTorch/XLA image.
    * At the bottom of the **OS Images** tab select the **Debian GNU/Linux 9 Stretch + PyTorch/XLA** image.
    * Chose an appropriate dist size based on your dataset and click **Select**.
    * Click **Create** to create the instance.


2. SSH into VM and activate the conda environment you wish to use. Each release (e.g.: `0.1`, `0.5`, `nightly`) is a separate conda environment.

    ```Shell
    (vm)$ export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    (vm)$ conda env list
    # conda environments:
    #
    base                  *  /anaconda3
    torch-xla-0.1              /anaconda3/envs/torch-xla-0.1
    torch-xla-0.5              /anaconda3/envs/torch-xla-0.5
    torch-xla-nightly          /anaconda3/envs/torch-xla-nightly

    (vm)$ conda activate torch-xla-0.5
    (torch-xla-0.5)$ cd /usr/share/torch-xla-0.5/pytorch/xla
    (torch-xla-0.5)$ python test/test_train_mnist.py
    ```

    To update the wheels `torch` and `torch_xla` to the latest nightly
    distribution (only updates your `torch-xla-nightly` conda env), run:
    ```Shell
    (vm)$ cd /usr/share/torch-xla-nightly/pytorch/xla
    (vm)$ . ./scripts/update_nightly_torch_wheels.sh
    ```

---

# How to Run on TPU Pods (distributed training)

Whereas the previous section focused on training on a single TPU node,
this section discusses distributed training in TPU Pods.

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

## Create your instance group
1. Create an instance template.
* During creation, make sure to go to section "Identity and API access" → "Access Scopes" and select "Allow full access to all Cloud APIs".
* If you have already have a VM instance running that you used to train PyTorch/TPU workloads and want to use that exact setup for distributed training: [instructions](https://cloud.google.com/compute/docs/instance-templates/create-instance-templates#based-on-existing-instance).
* Or, you can create an instance template using the PyTorch/XLA VM image we provide: [instructions](https://cloud.google.com/compute/docs/instance-templates/create-instance-templates#creating_a_new_instance_template).
2. Create an instance group to drive the TPU pod.
* This instance group is where all the input pipeline happens and where we feed all the tensors into the TPUs for training.
* Use the instance template created in step (1) to create your instance group.
* Make sure to (a) create the instance group in a single zone (same zone as the TPU Pod you'll create), (b) no autoscaling or health-checks, (c) number of instances (size of instance group) should be number of cores / 8 (ex. for a v3-32 you'd create an instance group of size 32/8 = 4).
* Here are the instructions for creating an instance group: [instructions](https://cloud.google.com/compute/docs/instance-groups/creating-groups-of-managed-instances#create_managed_group).

## Create your TPU Pod
1. [Create](https://pantheon.corp.google.com/compute/tpus) a TPU pod (same as creating regular TPUs, just select more cores when selecting TPU type).
* Make sure that the TPU is in the same zone as the instance group.
* Make sure that the size of your instance group follows: # instances in group = number of TPU cores / 8.

## Start distributed training
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

## List of VMs
If you up to not use an [instance group](#create-your-instance-group), you can decide to use a list of VM instances that you may have already created (or can create individually). Make sure that you create all the VM instances in the same zone as the TPU node, and also make sure that the VMs have the same configuration (datasets, VM size, disk size, etc.). Then you can [start distributed training](#start-distributed-training) after creating your TPU pod. The difference is in the `python -m torch_xla.distributed.xla_dist` command. For example, to use a list of VMs run the following command (ex. conda with v3-32):
```
(torch-xla-nightly)$ cd /usr/share/torch-xla-nightly/pytorch/xla
(torch-xla-nightly)$ python -m torch_xla.distributed.xla_dist --tpu=$TPU_POD_NAME --vm $VM1 --vm $VM2 --vm $VM3 --vm $VM4 --conda-env=torch-xla-nightly --env=XLA_USE_BF16=1 -- python test/test_train_imagenet.py --fake_data
```

To learn more about TPU Pods check out this [blog
post](https://cloud.google.com/blog/products/ai-machine-learning/googles-scalable-supercomputers-for-machine-learning-cloud-tpu-pods-are-now-publicly-available-in-beta).

---

# Running on Colab

You can also run your models on [Colab](https://github.com/pytorch/xla/tree/master/contrib/colab). However, do note that performance may be severely impacted when running on Colab compared to creating your own VM and TPU pair and there can be some bugs on Colab environment that may have already been fixed.

---

# Build Manually

Please note that we have nightly releases available so users usually don't have to build manually. This is mainly for OSS contributors.
Please refer to [contribution guide](CONTRIBUTING.md) for instructions to build from source.

# Tests

To run the tests, follow __one__ of the options below:

* Run on local CPU using the XRT client:

  ```Shell
  export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  export XRT_WORKERS="localservice:0;grpc://localhost:40934"
  ```

  Select any free TCP port you prefer instead of 40934 (totally arbitrary).

* Run on Cloud TPU using the XRT client, set the XRT_TPU_CONFIG environment variable:

  ```Shell
  export XRT_TPU_CONFIG="tpu_worker;0;<IP of the TPU node>:8470"
  ```

Note that the IP of the TPU node can change if the TPU node is reset. If _PyTorch_
seem to hang at startup, verify that the IP of your TPU node is still the same of
the one you have configured.

If you are planning to be building from source and hence using the latest _PyTorch/TPU_ code base,
it is suggested for you to select the _Nightly_ builds when you create a Cloud TPU instance.

Then run `test/run_tests.sh` and `test/cpp/run_tests.sh` to verify the setup is working.

# PyTorch/XLA API And Best Practices

Please check out the [API Guideline](API_GUIDE.md) for the best practices to write models to run on TPU & TPU Pod devices.

# Troubleshooting

If you see bad performance when using PyTorch/XLA, please check out the [troubleshooting guide](TROUBLESHOOTING.md) for how to avoid common pitfalls and how to debug.

# Communication

We use github issues to communicate with users and open source contributors. Please file an issue for questions, bug reports, feature requests, install issues, RFCs, thoughts, etc.

# Contributing

Please refer to [contribution guide](CONTRIBUTING.md) for detailed instructions.
