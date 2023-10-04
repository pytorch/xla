# PyTorch/XLA

<b>Current CI status:</b>  ![GitHub Actions
status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)

Note: PyTorch/XLA r2.1 will be the last release with XRT available as a legacy
runtime. Our main release build will not include XRT, but it will be available
in a separate package.

PyTorch/XLA is a Python package that uses the [XLA deep learning
compiler](https://www.tensorflow.org/xla) to connect the [PyTorch deep learning
framework](https://pytorch.org/) and [Cloud
TPUs](https://cloud.google.com/tpu/). You can try it right now, for free, on a
single Cloud TPU VM with
[Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)!

Take a look at one of our [Kaggle
notebooks](https://github.com/pytorch/xla/tree/master/contrib/kaggle) to get
started:

* [Stable Diffusion with PyTorch/XLA
  2.0](https://github.com/pytorch/xla/blob/master/contrib/kaggle/pytorch-xla-2-0-on-kaggle.ipynb)
* [Distributed PyTorch/XLA
  Basics](https://github.com/pytorch/xla/blob/master/contrib/kaggle/distributed-pytorch-xla-basics-with-pjrt.ipynb)

## Getting Started

**PyTorch/XLA is now on PyPI!**

To install PyTorch/XLA a new TPU VM:

```
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

To update your existing training loop, make the following changes:

```diff
-import torch.multiprocessing as mp
+import torch_xla.core.xla_model as xm
+import torch_xla.distributed.parallel_loader as pl
+import torch_xla.distributed.xla_multiprocessing as xmp

 def _mp_fn(index):
   ...

+  # Move the model paramters to your XLA device
+  model.to(xm.xla_device())
+
+  # MpDeviceLoader preloads data to the XLA device
+  xla_train_loader = pl.MpDeviceLoader(train_loader, xm.xla_device())

-  for inputs, labels in train_loader:
+  for inputs, labels in xla_train_loader:
     optimizer.zero_grad()
     outputs = model(inputs)
     loss = loss_fn(outputs, labels)
     loss.backward()
-    optimizer.step()
+
+    # `xm.optimizer_step` combines gradients across replicas
+    xm.optimizer_step()

 if __name__ == '__main__':
-  mp.spawn(_mp_fn, args=(), nprocs=world_size)
+  # xmp.spawn automatically selects the correct world size
+  xmp.spawn(_mp_fn, args=())
```

If you're using `DistributedDataParallel`, make the following changes:


```diff
 import torch.distributed as dist
-import torch.multiprocessing as mp
+import torch_xla.core.xla_model as xm
+import torch_xla.distributed.parallel_loader as pl
+import torch_xla.distributed.xla_multiprocessing as xmp
+import torch_xla.distributed.xla_backend

 def _mp_fn(rank, world_size):
   ...

-  os.environ['MASTER_ADDR'] = 'localhost'
-  os.environ['MASTER_PORT'] = '12355'
-  dist.init_process_group("gloo", rank=rank, world_size=world_size)
+  # Rank and world size are inferred from the XLA device runtime
+  dist.init_process_group("xla", init_method='xla://')
+
+  model.to(xm.xla_device())
+  # `gradient_as_bucket_view=True` required for XLA
+  ddp_model = DDP(model, gradient_as_bucket_view=True)

-  model = model.to(rank)
-  ddp_model = DDP(model, device_ids=[rank])
+  xla_train_loader = pl.MpDeviceLoader(train_loader, xm.xla_device())

-  for inputs, labels in train_loader:
+  for inputs, labels in xla_train_loader:
     optimizer.zero_grad()
     outputs = ddp_model(inputs)
     loss = loss_fn(outputs, labels)
     loss.backward()
     optimizer.step()

 if __name__ == '__main__':
-  mp.spawn(_mp_fn, args=(), nprocs=world_size)
+  xmp.spawn(_mp_fn, args=())
```

Additional information on PyTorch/XLA, including a description of its semantics
and functions, is available at [PyTorch.org](http://pytorch.org/xla/). See the
[API Guide](API_GUIDE.md) for best practices when writing networks that run on
XLA devices (TPU, GPU, CPU and...).

Our comprehensive user guides are available at:

[Documentation for the latest release](https://pytorch.org/xla)

[Documentation for master branch](https://pytorch.org/xla/master)


## PyTorch/XLA tutorials

* [Cloud TPU VM
  quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
* [Cloud TPU Pod slice
  quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
* [Profiling on TPU
  VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
* [GPU guide](docs/gpu.md)

## Available docker images and wheels

### Python packages

PyTorch/XLA releases starting with version r2.1 will be available on PyPI. You
can now install the main build with `pip install torch_xla`. To also install the
Cloud TPU plugin, install the optional `tpu` dependencies:

```
pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

GPU, XRT (legacy runtime), and nightly builds are available in our public GCS
bucket.

| Version | Cloud TPU VMs Wheel |
| --- | ----------- |
| 2.1 (CUDA 12.0 + Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.0/torch_xla-2.1.0-cp38-cp38-manylinux_2_28_x86_64.whl` |
| 2.1 (XRT + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/xrt/tpuvm/torch_xla-2.1.0%2Bxrt-cp310-cp310-manylinux_2_28_x86_64.whl` |
| nightly (Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl` |
| nightly (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl` |

<details>

<summary>older versions</summary>

| Version | Cloud TPU VMs Wheel |
|---------|-------------------|
| 2.0 (Python 3.8) | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl` |
| 1.13 | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.13-cp38-cp38-linux_x86_64.whl` |
| 1.12 | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.12-cp38-cp38-linux_x86_64.whl` |
| 1.11 | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.11-cp38-cp38-linux_x86_64.whl` |
| 1.10 | `https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.10-cp38-cp38-linux_x86_64.whl` |

<br/>

Note: For TPU Pod customers using XRT (our legacy runtime), we have custom
wheels for `torch`, `torchvision`, and `torch_xla` at
`https://storage.googleapis.com/tpu-pytorch/wheels/xrt`.

| Package | Cloud TPU VMs Wheel (XRT on Pod, Legacy Only) |
| --- | ----------- |
| torch_xla | `https://storage.googleapis.com/tpu-pytorch/wheels/xrt/torch_xla-2.0-cp38-cp38-linux_x86_64.whl` |
| torch | `https://storage.googleapis.com/tpu-pytorch/wheels/xrt/torch-2.0-cp38-cp38-linux_x86_64.whl` |
| torchvision | `https://storage.googleapis.com/tpu-pytorch/wheels/xrt/torchvision-2.0-cp38-cp38-linux_x86_64.whl` |

<br/>

| Version | GPU Wheel + Python 3.8 |
| --- | ----------- |
| 2.0 + CUDA 11.8 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/118/torch_xla-2.0-cp38-cp38-linux_x86_64.whl` |
| 2.0 + CUDA 11.7 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/117/torch_xla-2.0-cp38-cp38-linux_x86_64.whl` |
| 1.13 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-1.13-cp38-cp38-linux_x86_64.whl` |
| nightly + CUDA 12.0 >= 2023/06/27| `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.0/torch_xla-nightly-cp38-cp38-linux_x86_64.whl` |
| nightly + CUDA 11.8 <= 2023/04/25| `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/118/torch_xla-nightly-cp38-cp38-linux_x86_64.whl` |
| nightly + CUDA 11.8 >= 2023/04/25| `https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/11.8/torch_xla-nightly-cp38-cp38-linux_x86_64.whl` |

<br/>

| Version | GPU Wheel + Python 3.7 |
| --- | ----------- |
| 1.13 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-1.13-cp37-cp37m-linux_x86_64.whl` |
| 1.12 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-1.12-cp37-cp37m-linux_x86_64.whl` |
| 1.11 | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-1.11-cp37-cp37m-linux_x86_64.whl` |
| nightly | `https://storage.googleapis.com/tpu-pytorch/wheels/cuda/112/torch_xla-nightly-cp37-cp37-linux_x86_64.whl` |

<br/>

| Version | Colab TPU Wheel |
| --- | ----------- |
| 2.0 | `https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl` |

You can also add `+yyyymmdd` after `torch_xla-nightly` to get the nightly wheel
of a specified date. To get the companion pytorch and torchvision nightly wheel,
replace the `torch_xla` with `torch` or `torchvision` on above wheel links.

#### Installing libtpu (before PyTorch/XLA 2.0)

For PyTorch/XLA release r2.0 and older and when developing PyTorch/XLA, install
the `libtpu` pip package with the following command:

```
pip3 install torch_xla[tpuvm]
```

This is only required on Cloud TPU VMs.

</details>

### Docker

| Version | Cloud TPU VMs Docker |
| --- | ----------- |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_tpuvm` |
| 2.0 | `gcr.io/tpu-pytorch/xla:r2.0_3.8_tpuvm` |
| 1.13 | `gcr.io/tpu-pytorch/xla:r1.13_3.8_tpuvm` |
| nightly python | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm` |

<br/>

| Version | GPU CUDA 12.0 Docker |
| --- | ----------- |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_12.0` |
| nightly | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.0` |
| nightly at date | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.0_YYYYMMDD` |

<br/>

| Version | GPU CUDA 11.8 + Docker |
| --- | ----------- |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_cuda_11.8` |
| 2.0 | `gcr.io/tpu-pytorch/xla:r2.0_3.8_cuda_11.8` |
| nightly | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_11.8` |
| nightly at date | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_11.8_YYYYMMDD` |

<br/>

<details>

<summary>older versions</summary>

| Version | GPU CUDA 11.7 + Docker |
| --- | ----------- |
| 2.0 | `gcr.io/tpu-pytorch/xla:r2.0_3.8_cuda_11.7` |

<br/>

| Version | GPU CUDA 11.2 + Docker |
| --- | ----------- |
| 1.13 | `gcr.io/tpu-pytorch/xla:r1.13_3.8_cuda_11.2` |

<br/>

| Version | GPU CUDA 11.2 + Docker |
| --- | ----------- |
| 1.13 | `gcr.io/tpu-pytorch/xla:r1.13_3.7_cuda_11.2` |
| 1.12 | `gcr.io/tpu-pytorch/xla:r1.12_3.7_cuda_11.2` |

</details>

To run on [compute instances with
GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus).

## Troubleshooting

If PyTorch/XLA isn't performing as expected, see the [troubleshooting
guide](TROUBLESHOOTING.md), which has suggestions for debugging and optimizing
your network(s).

## Providing Feedback

The PyTorch/XLA team is always happy to hear from users and OSS contributors!
The best way to reach out is by filing an issue on this Github. Questions, bug
reports, feature requests, build issues, etc. are all welcome!

## Contributing

See the [contribution guide](CONTRIBUTING.md).

## Disclaimer

This repository is jointly operated and maintained by Google, Facebook and a
number of individual contributors listed in the
[CONTRIBUTORS](https://github.com/pytorch/xla/graphs/contributors) file. For
questions directed at Facebook, please send an email to opensource@fb.com. For
questions directed at Google, please send an email to
pytorch-xla@googlegroups.com. For all other questions, please open up an issue
in this repository [here](https://github.com/pytorch/xla/issues).

## Additional Reads

You can find additional useful reading materials in
* [Performance debugging on Cloud TPU
  VM](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-tpu-vm-part-1)
* [Lazy tensor
  intro](https://pytorch.org/blog/understanding-lazytensor-system-performance-with-pytorch-xla-on-cloud-tpu/)
* [Scaling deep learning workloads with PyTorch / XLA and Cloud TPU
  VM](https://cloud.google.com/blog/topics/developers-practitioners/scaling-deep-learning-workloads-pytorch-xla-and-cloud-tpu-vm)
* [Scaling PyTorch models on Cloud TPUs with
  FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/)
