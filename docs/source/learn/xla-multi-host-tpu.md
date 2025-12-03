# Introduction to multi-host PyTorch/XLA on TPU Pod

In this tutorial you will learn the details on how to scale PyTorch/XLA computations to TPU devices on multiple hosts, e.g. when running on a Cloud TPU Pod.

- **All hosts (usually) run the same Python script.** We write the Python code almost exactly the same as for a single host TPU — just run multiple instances of it and PyTorch/XLA takes care of the rest.


```{figure} https://docs.jax.dev/en/latest/_images/mcjax_overview.png
:alt: Illustration of a multi-host TPU pod. Each host in the pod is attached via PCI to a board of four TPU chips. The TPUs chips themselves are connected via high-speed inter-chip interconnects.

Illustration of a multi-host TPU pod. Each host in the pod (green) is attached
via PCI to a board of four TPU chips (blue). The TPUs chips themselves are
connected via high-speed inter-chip interconnects (ICI). PyTorch/XLA code runs on
each host, e.g. via ssh. The PyTorch/XLA processes on each host are aware of each other,
allowing you to orchestrate computation across the entire pods' worth of chips.
[Source](https://docs.jax.dev/en/latest/multi_process.html)
```

Before diving into the code and the commands to execute, let us introduce some terminology. The section below is an adapted version of [JAX multi-host tutorial](https://docs.jax.dev/en/latest/multi_process.html#terminology).

We sometimes call each Python process running PyTorch/XLA computations a _controller_ or a _host_, but the terms are essentially synonymous.

Each process (or host) has a set of **local devices**, meaning it can transfer data to and from those devices' memories and run computation on those devices without involving any other processes. The local devices are usually physically attached to the process's corresponding host, e.g. via PCI. A device can only be local to one process; that is, the local device sets are disjoint. The number of a process's local devices can be queried by evaluating {func}`torch_xla.runtime.addressable_runtime_device_count()`. We sometimes use the term **addressable** to mean the same thing as local.

```{figure} https://docs.jax.dev/en/latest/_images/controller_and_local_devices.png
:alt: Illustration of how a process/controller and local devices fit into a larger multi-host cluster. The "global devices" are all devices in the cluster.

Illustration of how a process/controller and local devices fit into a larger
multi-host cluster. The "global devices" are all devices in the cluster.
[Source](https://docs.jax.dev/en/latest/multi_process.html)
```

The devices across all processes are called the **global devices**. The number of global devices can be queried by {func}`torch_xla.runtime.global_runtime_device_count()`.


<!-- We often use the terms **global** and **local** to describe process-spanning and process-local concepts in general. For example, a "local tensor" could be a torch tensor that's only visible to a single process, compared to a PyTorch/XLA "global array" is conceptually visible to all processes. -->

## MNIST example

In this section we will use a simple script training a model on MNIST dataset using the SPMD model of computation and run it on a Google Cloud TPU Pod. Additionally, the training code sets up the profiler to monitor the performances per host.

### Google Cloud tools setup

We first need to install the `gcloud` CLI. The [official guide](https://cloud.google.com/sdk/docs/install) has full installation instructions. Below we provide the commands for Linux/Ubuntu:

<details>

<summary>
How to install the `gcloud` CLI on Linux/Ubuntu
</summary>

```bash
# Assuming root user with sudo rights
apt-get update && apt-get install -y curl ssh --no-install-recommends

curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz

tar -xf google-cloud-cli-linux-x86_64.tar.gz && rm -R google-cloud-cli-linux-x86_64.tar.gz && mv google-cloud-sdk /tmp/google-cloud-sdk

/tmp/google-cloud-sdk/install.sh
```

</details>

Next, we need to run the `gcloud` configuration command by choosing the project and compute zone:
```bash
gcloud init

# To verify the compute zone:
gcloud config get compute/zone
# if we use TPUs v4, the zone should be us-central2-b
```

#### (Optional) GCS Fuse tool

We can optionally install the `gcsfuse` tool to be able to mount Google Cloud Storage buckets. The [official guide](https://cloud.google.com/storage/docs/cloud-storage-fuse/install) has full installation instructions. Below we provide the commands for Linux/Ubuntu:

<details>

<summary>
How to install the `gcsfuse` CLI on Linux/Ubuntu
</summary>

```bash
# Assuming root user with sudo rights
export GCSFUSE_REPO=gcsfuse-noble
# export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get update
apt-get install -y fuse gcsfuse --no-install-recommends

# Check the installed version
# gcsfuse -v
```

</details>


### Python code

In this section we provide the Python code you can use to train a simple neural network model on MNIST dataset using the SPMD computation model. We will run this code on a multi-host TPU Pod, e.g. [v8-32](https://cloud.google.com/tpu/docs/v4).

The dataset will be automatically downloaded on each host. For more realistic use-cases, the training data can be fetched from a GCS bucket, for example, mounted as a folder to the filesystem using gcsfuse. For more information, please check this [documentation](https://cloud.google.com/tpu/docs/storage-options).

The code snippet below contains the complete training script. Please save it into a `mnist_xla.py` file. We will use this file in the next section to copy to TPU VM.


```python
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# PyTorch/XLA specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr


os.environ["XLA_IR_DEBUG"] = "1"
os.environ["XLA_HLO_DEBUG"] = "1"

# Enable the SPMD
xr.use_spmd()

# Declare meshes:
num_devices = xr.global_runtime_device_count()
device_ids = torch.arange(num_devices)
conv_mesh_shape = (num_devices // 2, 2, 1, 1)
conv_mesh = xs.Mesh(device_ids, conv_mesh_shape, ("data", "dim1", "dim2", "dim3"))

linear_mesh_shape = (num_devices // 2, 2)
linear_mesh = xs.Mesh(device_ids, linear_mesh_shape, ("data", "model"))
host_id = xr.process_index()

# Define the CNN Model
# We define the model weights sharding directly inside the model.
# Before marking the sharding of model weights, they should be put to
# XLA device.
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1).to(xm.xla_device())
        xs.mark_sharding(self.conv1.weight, conv_mesh, ("data", None, None, None))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1).to(xm.xla_device())
        xs.mark_sharding(self.conv2.weight, conv_mesh, ("data", None, None, None))

        self.fc1 = nn.Linear(9216, 128).to(xm.xla_device())
        xs.mark_sharding(self.fc1.weight, linear_mesh, ("data", None))

        self.fc2 = nn.Linear(128, 10).to(xm.xla_device())
        xs.mark_sharding(self.fc2.weight, linear_mesh, ("data", "model"))

    def forward(self, x):
        with xp.Trace("forward"):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)


def train_mnist():
    # Training parameters
    epochs = 2
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 256
    seed = 12

    torch.manual_seed(seed)

    # 1. Acquire the XLA device
    device = torch_xla.device()
    print(host_id, f"Running on XLA device: {device}")

    # Load MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # 2. Initialize the model and move it to the XLA device
    model = MNISTNet().to(device)

    # Define loss function and optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print(host_id, "Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with torch_xla.step():
                with xp.Trace("train_step_data_prep_and_forward"):
                    optimizer.zero_grad()

                    # 3. Move data and target to the XLA device
                    data, target = data.to(device), target.to(device)

                    # 4. Shard input
                    xs.mark_sharding(data, conv_mesh, ("data", "dim1", None, None))

                    output = model(data)

                with xp.Trace("train_step_loss_and_backward"):
                    loss = loss_fn(output, target)
                    loss.backward()

                with xp.Trace("train_step_optimizer_step_host"):
                    optimizer.step()

            torch_xla.sync()

            if batch_idx % 100 == 0:
                print(
                    host_id,
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    print(host_id, "Training finished!")


if __name__ == "__main__":
    assert (
        len(sys.argv) == 2
    ), "Usage: python mnist_xla.py /path/to/shared/output_folder"
    output_folder = sys.argv[1]
    output_folder = os.path.join(output_folder, f"proc_{host_id}")

    server = xp.start_server(9012)
    xp.start_trace(output_folder)
    train_mnist()
    xp.stop_trace()

```

Let's discuss the details of this training script. As we will see below when we run it on multiple TPUs, we will send the same command over SSH to all workers (e.g. to 8 workers if we are using v4-32): `python mnist_xla.py`.
Each worker executes in parallel the same Python commands and synchronizes with other workers on PyTorch/XLA API commands, e.g. `xr.global_runtime_device_count()`.

Before calling the training method `train_mnist()` we start the profiler:
```python
import torch_xla.debug.profiler as xp

server = xp.start_server(9012)
xp.start_trace(output_folder)
train_mnist()
xp.stop_trace()
```
and we mark parts of the training step with `xp.Trace` to create trace events, for example:
```python
with xp.Trace("train_step_data_prep_and_forward"):
    optimizer.zero_grad()
    data, target = data.to(device), target.to(device)
    xs.mark_sharding(data, conv_mesh, ("data", "dim1", None, None))
    output = model(data)
```

Note: writing traces files from multiple workers to a shared folder (e.g. a bucket mounted with `gcsfuse` as in our example) can present certain challenges. Namely, when multiple workers try to create the same folder, `gcsfuse` can raise a "Permission denied" error. To avoid these issues we force workers to create subfolders and write traces into different folders:
```python
# output_folder should exist
output_folder = os.path.join(output_folder, f"proc_{host_id}")
```

We enable SPMD computation model with `xr.use_spmd()`, and sharded tensors will be split between the workers according to the defined meshes:
```python
# Global meshes definition
num_devices = xr.global_runtime_device_count()
device_ids = torch.arange(num_devices)
conv_mesh_shape = (num_devices // 2, 2, 1, 1)
conv_mesh = xs.Mesh(device_ids, conv_mesh_shape, ("data", "dim1", "dim2", "dim3"))
linear_mesh_shape = (num_devices // 2, 2)
linear_mesh = xs.Mesh(device_ids, linear_mesh_shape, ("data", "model"))
```

The dataloader in this script is defined using `torch.utils.data.DataLoader` and the batches in the training step with shape `(B, 1, H, W)` are sharded across the batch dimension `B` on `D1 = num_devices // 2` devices. In other words, each worker keeps a local shard of shape `(B // D1, 1, H, W)` of the loaded data to form a global tensor of the training data batch of shape `(B, 1, H, W)`. The following code is responsible for this sharding:
```python
# Train step
data, target = data.to(device), target.to(device)
xs.mark_sharding(data, conv_mesh, ("data", "dim1", None, None))
```
The model contains convolutional and fully-connected layers and we also shard them. The convolution's weights of shape `(OutFeat, InFeat, K, K)` are sharded such that a single worker has a local shard of shape `(OutFeat // D1, InFeat, K, K)`.
The fully-connected layer `fc1` is sharded similarly to the convolutions and `fc2` are sharded over two dimensions as following `(OutFeat // D1, InFeat // 2)`. In this script, the model's sharding is defined directly inside the constructor method:
```python
class MNISTNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1).to(xm.xla_device())
        xs.mark_sharding(self.conv1.weight, conv_mesh, ("data", None, None, None))
        ...
        self.fc2 = nn.Linear(128, 10).to(xm.xla_device())
        xs.mark_sharding(self.fc2.weight, linear_mesh, ("data", "model"))
```

### Run on TPU Pod

Let's first create a cloud storage for the output profiler logs:
```bash
export OUTPUT_BUCKET_NAME=torch-xla-xprof-outputs
gcloud storage buckets create gs://${OUTPUT_BUCKET_NAME}
```

Next, we create a multi-host TPU VM:
```bash
# TPU setup
export ACCELERATOR_TYPE=v4-32
export RUNTIME_VERSION=tpu-ubuntu2204-base
export TPU_NAME=torch-xla-multihost-example-${ACCELERATOR_TYPE}

# We use docker image to run the code
export DOCKER_IMAGE=us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_tpuvm

# Create a start-up script executed on all TPU VMs of the pod on the start-up:
cat << EOF > example_startup.sh
#!/bin/bash
set -eux

# get torch_xla docker image
docker pull $DOCKER_IMAGE

# Install gcsfuse
export GCSFUSE_REPO=gcsfuse-\`lsb_release -c -s\`
echo "deb https://packages.cloud.google.com/apt \$GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get update
apt-get install -y fuse gcsfuse --no-install-recommends

# Mount GCS bucket
mkdir -p /root/logs
gcsfuse ${OUTPUT_BUCKET_NAME} /root/logs

EOF

# Create a spot TPU VM:
gcloud compute tpus tpu-vm create $TPU_NAME --spot \
    --accelerator-type=$ACCELERATOR_TYPE \
    --version=$RUNTIME_VERSION \
    --metadata-from-file=startup-script=example_startup.sh

```

Once the TPU VM is ready, we copy the previously created `mnist_xla.py` file to all
workers of the TPU VM. We then define the training command inside the docker container we want to run on all workers.

```bash
# Create /root/example/ folder on all workers and copy mnist_xla.py
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command "mkdir -p /root/example/"
gcloud compute tpus tpu-vm scp --worker=all mnist_xla.py $TPU_NAME:/root/example/mnist_xla.py

# Define the command to run on all workers
# - we call the container: example-run
# - we mount /root folder to the container and set /root/example as current working directory
docker_run_args="--name=example-run --rm --privileged --net=host --ipc=host -e PJRT_DEVICE=TPU -v /root:/root -w /root/example"

docker_run_cmd="docker run ${docker_run_args} $DOCKER_IMAGE"

# In the start-up script we mounted /root/logs to our cloud bucket and we use this folder to write the logs:
full_command="cd /root/example && ${docker_run_cmd} python -u mnist_xla.py /root/logs/"

# Run the command and write the stdout into mnist_xla.$ACCELERATOR_TYPE.log file:
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command "${full_command}" &> mnist_xla.$ACCELERATOR_TYPE.log
```

A successful run will write the following content, for example using v4-32:
```
Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 4 workers.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
...
1 Running on XLA device: xla:0
3 Running on XLA device: xla:0
2 Running on XLA device: xla:0
0 Running on XLA device: xla:0
1 Starting training...
2 Starting training...
3 Starting training...
0 Starting training...
2 Train Epoch: 1 [0/60000 (0%)]	Loss: 2.305365
3 Train Epoch: 1 [0/60000 (0%)]	Loss: 2.305365
1 Train Epoch: 1 [0/60000 (0%)]	Loss: 2.305365
0 Train Epoch: 1 [0/60000 (0%)]	Loss: 2.305365
...
2 Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.165178
3 Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.165178
0 Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.165178
1 Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.165178
1 Training finished!
3 Training finished!
2 Training finished!
0 Training finished!
```

#### Profiler logs in TensorBoard

We can inspect the profiler logs using TensorBoard. We will spin up an instance of the TensorBoard on the local machine and read the logs from the bucket mounted using `gcsfuse`. Alternatively, one can spin up the TensorBoard instance on the TPU VM.

- Install TensorBoard:
```bash
pip install -U tensorboard-plugin-profile tensorboard
```

- Mount the logs from the bucket:
```bash
mkdir -p remote_logs
gcsfuse ${OUTPUT_BUCKET_NAME} remote_logs
```

- Start TensorBoard:
```bash
tensorboard --logdir=remote_logs/ --port 7007
```

#### Troubleshooting

We can execute commands on all workers together or on a single worker:
```bash
# All workers:
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command "docker ps -a"
# worker 0:
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=0 --command "docker ps -a"
```

To stop a running training on all workers, we can simply stop the docker container (our container is called "example-run"):
```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=all --command "docker stop example-run"
```

We can also connect to a single worker and inspect its filesystem and execute some commands. However, please note that the execution of PyTorch/XLA methods on a single worker may hang due to internal collective operations.

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --worker=0
```


### Clean up

To avoid incurring charges to your Google Cloud account for the resources used on this tutorial, follow these steps.

Delete the TPU VMs:
```bash
yes | gcloud compute tpus tpu-vm delete $TPU_NAME --async

gcloud compute tpus tpu-vm list
```
