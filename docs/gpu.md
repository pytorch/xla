# How to run with PyTorch/XLA:GPU

PyTorch/XLA enables PyTorch users to utilize the XLA compiler which supports accelerators including TPU, GPU, and CPU. This doc will go over the basic steps to run PyTorch/XLA on a nvidia GPU instances.

## Create a GPU instance

You can either use a local machine with GPU attached or a GPU VM on the cloud. For example in Google Cloud you can follow this [doc](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus) to create the GPU VM.

## Environment Setup

### Docker
Pytorch/XLA currently publish prebuilt docker images and wheels with cuda11.7/8 and python 3.8. We recommend users to create a docker container with corresponding config. For a full list of docker images and wheels, please refer to [this doc](https://github.com/pytorch/xla#available-docker-images-and-wheels).
```
sudo docker pull us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_11.8
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent    software-properties-common
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
sudo docker run --shm-size=16g --net=host --gpus all -it -d us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_11.8 bin/bash
sudo docker exec -it $(sudo docker ps | awk 'NR==2 { print $1 }') /bin/bash
```

Note that you need to restart the docker to make gpu devices visible in the docker container. After logging into the docker, you can use `nvidia-smi` to verify the device is setup correctly.

```
(pytorch) root@20ab2c7a2d06:/# nvidia-smi
Thu Dec  8 06:24:29 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   36C    P0    38W / 300W |      0MiB / 16384MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```

### Wheel
```
pip3 install torch==2.0
pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/cuda/117/torch_xla-2.0-cp38-cp38-linux_x86_64.whl
```

## Run a simple model
In order to run below examples, you need to clone the pytorch/xla repo to access the imagenet example(We already clone it in our docker).

```
(pytorch) root@20ab2c7a2d06:/# export GPU_NUM_DEVICES=1 PJRT_DEVICE=CUDA
(pytorch) root@20ab2c7a2d06:/# git clone --recursive https://github.com/pytorch/xla.git
(pytorch) root@20ab2c7a2d06:/# python xla/test/test_train_mp_imagenet.py --fake_data
==> Preparing data..
Epoch 1 train begin 06:12:38
| Training Device=xla:0/0 Epoch=1 Step=0 Loss=6.89059 Rate=2.82 GlobalRate=2.82 Time=06:13:23
| Training Device=xla:0/0 Epoch=1 Step=20 Loss=6.79297 Rate=117.16 GlobalRate=45.84 Time=06:13:36
| Training Device=xla:0/0 Epoch=1 Step=40 Loss=6.43628 Rate=281.16 GlobalRate=80.49 Time=06:13:43
| Training Device=xla:0/0 Epoch=1 Step=60 Loss=5.83108 Rate=346.88 GlobalRate=108.82 Time=06:13:49
| Training Device=xla:0/0 Epoch=1 Step=80 Loss=4.99023 Rate=373.62 GlobalRate=132.43 Time=06:13:56
| Training Device=xla:0/0 Epoch=1 Step=100 Loss=3.92699 Rate=384.33 GlobalRate=152.40 Time=06:14:02
| Training Device=xla:0/0 Epoch=1 Step=120 Loss=2.68816 Rate=388.35 GlobalRate=169.49 Time=06:14:09
```
## AMP (AUTOMATIC MIXED PRECISION)
AMP is very useful on GPU training and PyTorch/XLA reuse Cuda's AMP rule. You can checkout our [mnist example](https://github.com/pytorch/xla/blob/master/test/test_train_mp_mnist_amp.py) and [imagenet example](https://github.com/pytorch/xla/blob/master/test/test_train_mp_imagenet_amp.py). Note that we also used a modified version of [optimizers](https://github.com/pytorch/xla/tree/master/torch_xla/amp/syncfree) to avoid the additional sync between device and host.

## Develop PyTorch/XLA on a GPU instance (build PyTorch/XLA from source with GPU support)

1. Inside a GPU VM, create a docker container from a development docker image. For example:

```
sudo docker pull us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:3.8_cuda_11.8
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent    software-properties-common
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
sudo docker run --shm-size=16g --net=host --gpus all -it -d us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/development:3.8_cuda_11.8
sudo docker exec -it $(sudo docker ps | awk 'NR==2 { print $1 }') /bin/bash
```

2. Build PyTorch and PyTorch/XLA from source.

```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
USE_CUDA=0 python setup.py install

git clone https://github.com/pytorch/xla.git
cd xla
XLA_CUDA=1 python setup.py install
```

3. Verify if PyTorch and PyTorch/XLA have been installed successfully.

If you can run the test in the section
[Run a simple model](#run-a-simple-model) successfully, then PyTorch and
PyTorch/XLA should have been installed successfully.
