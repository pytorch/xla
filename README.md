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

Please find tutorials on our [GitHub page](https://github.com/pytorch/xla) for the latest release.

## Installation

### TPU

To install PyTorch/XLA a new TPU VM:

```
pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

### GPU Plugin (beta)

PyTorch/XLA now provides GPU support through a plugin package similar to `libtpu`:

```
pip torch~=2.3.0 torch_xla~=2.3.0 https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla_cuda_plugin-2.3.0-py3-none-any.whl
```

To use the plugin, set `XLA_REGISTER_INSTALLED_PLUGINS=1` or call `torch_xla.experimental.plugins.use_dynamic_plugins()` in your script.

The CUDA plugin is considered beta for the 2.3 release, and it will become standard in the near future. For more information about device plugins, see issue #6242.

For a list of all `torch_xla` packages with statically-linked CUDA support, see our [main README](https://github.com/pytorch/xla/blob/master/README.md).

## Getting Started

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

 def _mp_fn(rank):
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
XLA devices (TPU, CUDA, CPU and...).

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

For all builds and all versions of `torch-xla`, see our main [GitHub
README](https://github.com/pytorch/xla).

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
