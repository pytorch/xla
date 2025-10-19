# PyTorch/XLA

<b>Current CI status:</b>  ![GitHub Actions
status](https://github.com/pytorch/xla/actions/workflows/build_and_test.yml/badge.svg)

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

## Installation

### TPU

To install PyTorch/XLA stable build in a new TPU VM:
Note: Builds are available for Python 3.8 to 3.11; please use one of the supported versions.

```sh
# - for venv
# python3.11 -m venv py311
# - for conda
# conda create -n py311 python=3.11

pip install torch==2.8.0 'torch_xla[tpu]==2.8.0'

# Optional: if you're using custom kernels, install pallas dependencies
pip install --pre torch_xla[pallas] --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
**As of 07/16/2025 and starting from Pytorch/XLA 2.8 release, PyTorch/XLA will 
provide nightly and release wheels for Python 3.11 to 3.13**
To install PyTorch/XLA nightly build in a new TPU VM:

```sh
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
# Edit `cp310-cp310` to fit your desired Python version as needed
pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl' \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
```

### C++11 ABI builds
**As of 03/18/2025 and starting from Pytorch/XLA 2.7 release, C++11 ABI builds
are the default and we no longer provide wheels built with pre-C++11 ABI.**

In Pytorch/XLA 2.6, we'll provide wheels and docker images built with
two C++ ABI flavors: C++11 and pre-C++11. Pre-C++11 is the default to align with
PyTorch upstream, but C++11 ABI wheels and docker images have better lazy tensor
tracing performance.

To install C++11 ABI flavored 2.6 wheels (Python 3.10 example):

```sh
pip install torch==2.6.0+cpu.cxx11.abi \
  https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl \
  'torch_xla[tpu]' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://download.pytorch.org/whl/torch
```

The above command works for Python 3.10. We additionally have Python 3.9 and 3.11
wheels:

- 3.9: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp39-cp39-manylinux_2_28_x86_64.whl
- 3.10: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp310-cp310-manylinux_2_28_x86_64.whl
- 3.11: https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0%2Bcxx11-cp311-cp311-manylinux_2_28_x86_64.whl

To access C++11 ABI flavored docker image:

```
us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11
```

If your model is tracing bound (e.g. you see that the host CPU is busy tracing
the model while TPUs are idle), switching to the C++11 ABI wheels/docker images
can improve performance. Mixtral 8x7B benchmarking results on v5p-256, global
batch size 1024:

- Pre-C++11 ABI MFU: 33%
- C++ ABI MFU: 39%


## Github Doc Map

Our github contains many useful docs on working with different aspects of PyTorch XLA, here is a list of useful docs spread around our repository:

- [docs/source/learn](https://github.com/pytorch/xla/tree/master/docs/source/learn): docs for learning concepts associated with XLA, troubleshooting, pjrt, eager mode, and dynamic shape.
- [docs/source/accelerators](https://github.com/pytorch/xla/tree/master/docs/source/accelerators): references to `TPU` accelerator documents.
- [docs/source/perf](https://github.com/pytorch/xla/tree/master/docs/source/perf): documentation about performance specific aspects of PyTorch/XLA such as: `AMP`, `DDP`, `Dynamo`, Fori loop, `FSDP`, quantization, recompilation, and `SPMD`
- [docs/source/features](https://github.com/pytorch/xla/tree/master/docs/source/features): documentation on distributed torch, pallas, scan, and stable hlo.
- [docs/source/contribute](https://github.com/pytorch/xla/tree/master/docs/source/contribute): documents on setting up PyTorch for development, and guides for lowering operations.
- PJRT plugins:
  - [CPU](https://github.com/pytorch/xla/blob/master/plugins/cpu/README.md)
- [torchax/docs](https://github.com/pytorch/xla/tree/master/torchax/docs): torchax documents
  - [torchax/examples](https://github.com/pytorch/xla/tree/master/torchax/examples): torchax examples

## Getting Started

Following here are guides for two modes:
- Single process: one Python interpreter controlling a single TPU at a time
- Multi process: N Python interpreters are launched, corresponding to N TPUs
found on the system

Another mode is SPMD, where one Python interpreter controls all N TPUs found on
the system. Multi processing is more complex, and is not compatible with SPMD. This
tutorial does not dive into SPMD. For more on that, check our
[SPMD guide](https://github.com/pytorch/xla/blob/master/docs/source/perf/spmd_basic.md).

### Simple single process

To update your exisitng training loop, make the following changes:

```diff
+import torch_xla

 def train(model, training_data, ...):
   ...
   for inputs, labels in train_loader:
+    with torch_xla.step():
       inputs, labels = training_data[i]
+      inputs, labels = inputs.to('xla'), labels.to('xla')
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
       optimizer.step()

+  torch_xla.sync()
   ...

 if __name__ == '__main__':
   ...
+  # Move the model paramters to your XLA device
+  model.to('xla')
   train(model, training_data, ...)
   ...
```

The changes above should get your model to train on the TPU.

### Multi processing

To update your existing training loop, make the following changes:

```diff
-import torch.multiprocessing as mp
+import torch_xla
+import torch_xla.core.xla_model as xm

 def _mp_fn(index):
   ...

+  # Move the model paramters to your XLA device
+  model.to('xla')

   for inputs, labels in train_loader:
+    with torch_xla.step():
+      # Transfer data to the XLA device. This happens asynchronously.
+      inputs, labels = inputs.to('xla'), labels.to('xla')
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
-      optimizer.step()
+      # `xm.optimizer_step` combines gradients across replicas
+      xm.optimizer_step(optimizer)

 if __name__ == '__main__':
-  mp.spawn(_mp_fn, args=(), nprocs=world_size)
+  # torch_xla.launch automatically selects the correct world size
+  torch_xla.launch(_mp_fn, args=())
```

If you're using `DistributedDataParallel`, make the following changes:


```diff
 import torch.distributed as dist
-import torch.multiprocessing as mp
+import torch_xla
+import torch_xla.distributed.xla_backend

 def _mp_fn(rank):
   ...

-  os.environ['MASTER_ADDR'] = 'localhost'
-  os.environ['MASTER_PORT'] = '12355'
-  dist.init_process_group("gloo", rank=rank, world_size=world_size)
+  # Rank and world size are inferred from the XLA device runtime
+  dist.init_process_group("xla", init_method='xla://')
+
+  model.to('xla')
+  ddp_model = DDP(model, gradient_as_bucket_view=True)

-  model = model.to(rank)
-  ddp_model = DDP(model, device_ids=[rank])

   for inputs, labels in train_loader:
+    with torch_xla.step():
+      inputs, labels = inputs.to('xla'), labels.to('xla')
       optimizer.zero_grad()
       outputs = ddp_model(inputs)
       loss = loss_fn(outputs, labels)
       loss.backward()
       optimizer.step()

 if __name__ == '__main__':
-  mp.spawn(_mp_fn, args=(), nprocs=world_size)
+  torch_xla.launch(_mp_fn, args=())
```

Additional information on PyTorch/XLA, including a description of its semantics
and functions, is available at [PyTorch.org](http://pytorch.org/xla/). See the
[API Guide](API_GUIDE.md) for best practices when writing networks that run on
XLA devices (TPU, CPU and...).

Our comprehensive user guides are available at:

[Documentation for the latest release](https://pytorch.org/xla)

[Documentation for master branch](https://pytorch.org/xla/master)


## PyTorch/XLA tutorials

* [Cloud TPU VM quickstart](https://cloud.google.com/tpu/docs/run-calculation-pytorch)
* [Cloud TPU Pod slice quickstart](https://cloud.google.com/tpu/docs/pytorch-pods)
* [Profiling on TPU VM](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)

## Reference implementations

The [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes)
repo. contains examples for training and serving many LLM and diffusion models.

## Available docker images and wheels

### Python packages

PyTorch/XLA releases starting with version r2.1 will be available on PyPI. You
can now install the main build with `pip install torch_xla`. To also install the
Cloud TPU plugin corresponding to your installed `torch_xla`, install the optional `tpu` dependencies after installing the main build with

```
pip install torch_xla[tpu]
```

TPU nightly builds are available in our public GCS bucket.

| Version | Cloud TPU Nightly Wheels |
| --- | ----------- |
| nightly (Python 3.11) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp311-cp311-linux_x86_64.whl` |
| nightly (Python 3.12) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |
| nightly (Python 3.13) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev-cp312-cp312-linux_x86_64.whl` |

#### Use nightly build

You can also add `yyyymmdd` like `torch_xla-2.9.0.devyyyymmdd` (or the latest dev version)
to get the nightly wheel of a specified date. Here is an example:

```
pip3 install torch==2.9.0.dev20250423+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.9.0.dev20250423-cp310-cp310-linux_x86_64.whl
```

The torch wheel version `2.9.0.dev20250423+cpu` can be found at https://download.pytorch.org/whl/nightly/torch/.

<details>

<summary>older versions</summary>

| Version | Cloud TPU VMs Wheel |
|---------|-------------------|
| 2.7 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.7.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.6 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.5 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.5.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.4 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.3 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.3.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.2 (Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.2.0-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 (XRT + Python 3.10) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/xrt/tpuvm/torch_xla-2.1.0%2Bxrt-cp310-cp310-manylinux_2_28_x86_64.whl` |
| 2.1 (Python 3.8) | `https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.1.0-cp38-cp38-linux_x86_64.whl` |

</details>

### Docker
NOTE: Since PyTorch/XLA 2.7, all builds will use the C++11 ABI by default
| Version | Cloud TPU VMs Docker |
| --- | ----------- |
| 2.7 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.7.0_3.10_tpuvm` |
| 2.6 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm` |
| 2.6 (C++11 ABI) | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.6.0_3.10_tpuvm_cxx11` |
| 2.5 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_tpuvm` |
| 2.4 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_tpuvm` |
| 2.3 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.3.0_3.10_tpuvm` |
| 2.2 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_tpuvm` |
| 2.1 | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.1.0_3.10_tpuvm` |
| nightly python | `us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm` |

To use the above dockers, please pass `--privileged --net host --shm-size=16G` along. Here is an example:
```bash
docker run --privileged --net host --shm-size=16G -it us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm /bin/bash
```

## Troubleshooting

If PyTorch/XLA isn't performing as expected, see the [troubleshooting
guide](docs/source/learn/troubleshoot.md), which has suggestions for debugging and optimizing
your network(s).

## Providing Feedback

The PyTorch/XLA team is always happy to hear from users and OSS contributors!
The best way to reach out is by filing an issue on this Github. Questions, bug
reports, feature requests, build issues, etc. are all welcome!

## Contributing

See the [contribution guide](CONTRIBUTING.md).

## Disclaimer

This repository is jointly operated and maintained by Google, Meta and a
number of individual contributors listed in the
[CONTRIBUTORS](https://github.com/pytorch/xla/graphs/contributors) file. For
questions directed at Meta, please send an email to opensource@fb.com. For
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

## Related Projects

* [OpenXLA](https://github.com/openxla)
* [HuggingFace](https://huggingface.co/docs/accelerate/en/basic_tutorials/tpu)
* [JetStream](https://github.com/google/JetStream-pytorch)
