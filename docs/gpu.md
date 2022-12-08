# How to run with PyTorch/XLA:GPU

PyTorch/XLA enables PyTorch users to utilize the XLA compiler which supports accelerators including TPU, GPU, CPU and … This doc will go over the basic steps to run PyTorch/XLA on a nvidia gpu instance

### Create a GPU instance
Pytorch/XLA currently publish prebuilt docker images and wheels with cuda11.2 and python 3.7/3.8. We recommend users to create a GPU instance with corresponding config. For a full list of docker images and wheels, please refer to this doc.

### Setup the docker environment
```
sudo docker pull gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.2
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent    software-properties-common
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
sudo docker run --gpus all -it -d gcr.io/tpu-pytorch/xla:nightly_3.7\8_cuda_11.2 bin/bash
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


### Run a simple model
```
(pytorch) root@20ab2c7a2d06:/# export GPU_NUM_DEVICES=1
(pytorch) root@20ab2c7a2d06:/# python pytorch/xla/test/test_train_mp_imagenet.py --fake_data
==> Preparing data..
Epoch 1 train begin 06:12:38
2022-12-08 06:13:12.452874: W      79 tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.cc:729] None of the algorithms provided by cuDNN heuristics worked; trying fallback algorithms.  Conv: (f32[128,256,28,28]{3,2,1,0}, u8[0]{0}) custom-call(f32[128,256,14,14]{3,2,1,0}, f32[3,3,256,256]{1,0,2,3}), window={size=3x3 stride=2x2 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convBackwardInput", backend_config="{\"conv_result_scale\":1,\"activation_mode\":\"0\",\"side_input_scale\":0}"
2022-12-08 06:13:13.780992: W      79 tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.cc:729] None of the algorithms provided by cuDNN heuristics worked; trying fallback algorithms.  Conv: (f32[128,128,56,56]{3,2,1,0}, u8[0]{0}) custom-call(f32[128,128,28,28]{3,2,1,0}, f32[3,3,128,128]{1,0,2,3}), window={size=3x3 stride=2x2 pad=1_1x1_1}, dim_labels=bf01_01io->bf01, custom_call_target="__cudnn$convBackwardInput", backend_config="{\"conv_result_scale\":1,\"activation_mode\":\"0\",\"side_input_scale\":0}"
| Training Device=xla:0/0 Epoch=1 Step=0 Loss=6.89059 Rate=2.82 GlobalRate=2.82 Time=06:13:23
| Training Device=xla:0/0 Epoch=1 Step=20 Loss=6.79297 Rate=117.16 GlobalRate=45.84 Time=06:13:36
| Training Device=xla:0/0 Epoch=1 Step=40 Loss=6.43628 Rate=281.16 GlobalRate=80.49 Time=06:13:43
| Training Device=xla:0/0 Epoch=1 Step=60 Loss=5.83108 Rate=346.88 GlobalRate=108.82 Time=06:13:49
| Training Device=xla:0/0 Epoch=1 Step=80 Loss=4.99023 Rate=373.62 GlobalRate=132.43 Time=06:13:56
| Training Device=xla:0/0 Epoch=1 Step=100 Loss=3.92699 Rate=384.33 GlobalRate=152.40 Time=06:14:02
| Training Device=xla:0/0 Epoch=1 Step=120 Loss=2.68816 Rate=388.35 GlobalRate=169.49 Time=06:14:09
```
