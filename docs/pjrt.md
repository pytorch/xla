# Experimental PjRt Runtime Support

_This document reflects the current state of PjRt support in current nightly
builds_. See the [same document on the r1.13 branch](https://github.com/pytorch/xla/blob/r1.13/docs/pjrt.md)
for the status in the latest stable release.

The PyTorch/XLA team is currently migrating from the currently-supported XRT
runtime to the [PjRt
runtime](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/pjrt)
used by [JAX](https://github.com/google/jax).

PjRt is available as an _experimental preview_ in PyTorch/XLA r1.13. The
PyTorch/XLA team will provide limited support on a best-effort basis during this
preview. If you encounter a bug with PjRt, please file an issue on GitHub with
the `runtime` tag.

**This preview is mainly targeted at TPU v4**. In most cases, we expect that you can
re-use your existing PyTorch/XLA code for TPU v4 with no changes. You may be able to
adapt your v2 or v3 workload to PjRt with some caveats (see below).

## Quickstart

To start using PjRt with PyTorch/XLA, all you need to do is set the
`PJRT_DEVICE` environment variable. If you're working on a TPU v2 or v3, keep
reading to learn about the differences between TPU v2 and v3 and v4.

### CPU

On any machine with PyTorch/XLA installed, you can run our MNIST example on CPU
like this:

```
PJRT_DEVICE=CPU python3 xla/test/test_train_mp_mnist.py --fake_data
```

### TPU

To create a new TPU with PyTorch/XLA r1.13 installed:

```
gcloud alpha compute tpus tpu-vm create $USER-pjrt --accelerator-type=v4-8 --version=tpu-vm-v4-pt-1.13 --zone=us-central2-b --project=$PROJECT
```

On a v4-8, you can run our ResNet50 example like this:

```
git clone --depth=1 --branch r1.13 https://github.com/pytorch/xla.git
PJRT_DEVICE=TPU python3 xla/test/test_train_mp_imagenet.py --fake_data --batch_size=256 --num_epochs=1
```

By default, PjRt will use all TPU chips. To use only one TPU chip, configure
`TPU_PROCESS_BOUNDS` and `TPU_VISIBLE_CHIPS`:

```
TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_CHIPS=0 PJRT_DEVICE=TPU python3 xla/test/test_train_mp_imagenet.py --fake_data --batch_size=256 --num_epochs=1
```

#### Pods

On TPU Pods, use `gcloud` to run your command on each TPU in parallel:

```
gcloud alpha compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command="git clone --depth=1 --branch r1.13 https://github.com/pytorch/xla.git"
gcloud alpha compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command="PJRT_DEVICE=TPU python3 xla/test/test_train_mp_imagenet.py --fake_data --batch_size=256 --num_epochs=1"
```

#### Docker

You can also use Docker to run your workload in a container with PyTorch/XLA
preinstalled:

```
export DOCKER_IMAGE=gcr.io/...

# Optional: authenticate docker if your image is in a private GCP repository
gcloud compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command "sudo gcloud auth configure-docker"

# Run your workload
gcloud compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command "sudo docker run --rm --privileged --net=host -e PJRT_DEVICE=TPU $DOCKER_IMAGE python pytorch/xla/test/test_train_mp_imagenet.py --fake_data"
```

Note that `docker run` requires privileged access to the host (`--privileged`)
to expose the TPU device to the container. Docker on TPU pods is only supported
with host networking `--net=host` at this time. See the [Cloud TPU documentation](https://cloud.google.com/tpu/docs/run-in-container)
for more information.

### GPU

*Warning: GPU support is still highly experimental!*

To use GPUs with PjRt, simply set `PJRT_DEVICE=GPU` and configure
`GPU_NUM_DEVICES` to the number of devices on the host. For example:

```
PJRT_DEVICE=GPU GPU_NUM_DEVICES=4 python3 xla/test/test_train_mp_imagenet.py --fake_data --batch_size=128 --num_epochs=1
```

Currently, only a single host is supported, and multi-host GPU cluster support
will be added in an future release.

#### Known Issues

The GPU integration has issues with replica groups in collectives (i.e. the
`group` parameter of the XLA collective ops). If the replica groups are
changed, there is a chance that the process will hang. For now, the
recommendation is to use a single replica group containing all devices, as is
the case in data parallel training.

## Key differences from XRT

Although in most cases we expect PjRt and XRT to work mostly interchangeably
from the end-user's perspective (especially on TPU v4), there are some subtle
differences that are important to keep in mind. Importantly, XRT was designed
around the TPU Node architecture, so it will always spawn a client and a server
process, even on TPU VMs. Thus, every batch of inputs has additional latency
from serializing and deserializing data.

PjRt uses the local device directly with no intermediate server process. In the
default configuration, PjRt will create one process per TPU chip, or 4 processes
per TPU host. See the [Cloud TPU documentation](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
for more information about TPU architecture.

- Performance gains are possible for workloads constrained by data transfer
  speeds.
- Under XRT, the server process is the only process that interacts with the TPU
  devices, and client processes don't have direct access to the TPU devices.
  When profiling a single-host TPU (e.g. v3-8 or v4-8), you would normally see 8
  device traces (one for each TPU core). With PjRt, each process has one chip,
  and a profile from that process will show only 2 TPU cores.
- For the same reason, profiling does not work on TPU Pods with XRT, because the
  server process runs independently from the user's model code. PjRt does not
  have that constraint, so it is possible to profile 2 TPU cores per process in
  a TPU Pod.
- PjRt only supports the TPU VM architecture and we have no plans to support the
  TPU Node architecture with PjRt.
- Runtime configuration is significantly simpler with PjRt. `xla_dist` is not
  required to run TPU Pod workloads. Instead, copy your code to each TPU host
  ([`gcloud compute tpus tpu-vm scp`](https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/tpu-vm/scp)) and run the code on each host in
  parallel (e.g. [`gcloud compute tpus tpu-vm ssh --workers=all
  --command="PJRT_DEVICE=TPU python run.py"`](https://cloud.google.com/sdk/gcloud/reference/alpha/compute/tpus/tpu-vm/ssh))
- `xm.rendezvous` has been reimplemented using XLA-native collective
  communication to enhance stability on large TPU pods. See below for more
  details.

### Changes to `xm.rendezvous`

_New in PyTorch/XLA r1.14 (nightly only)_

In practice, we found that running a single mesh master process was unreliable
on TPU pods with thousands of chips due to the number of inbound connections to
worker 0. A single client process timing out could cause a failure and force the
entire workload to restart.

Thus, we have reimplemented `xm.rendezvous` with native XLA collective
communication, which is much more stable and well-tested on large TPU pods. This
imposes two new constraints compared to the XRT implementation:

- Because the payload has to become part of the XLA graph, `xm.mark_step` is
  called both before and after the data is transferred. Calling `xm.rendezvous`
  in the middle of model code may force an unwanted compilation.
- Because XLA does not permit collective operations to run on a subset of
  workers, all workers must participate in the `rendezvous`.

If you require the old behavior of `xm.rendezvous` (i.e. communicating data
without altering the XLA graph and/or synchronizing a subset of workers),
consider using [`torch.distributed.barrier`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.barrier)
or [`torch.distributed.all_gather_object`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_object)
with a `gloo` process group. If you are also using the `xla` `torch.distributed`
backend, you can use `torch.new_group` to create a `gloo` subgroup. See [this
example](https://pytorch.org/docs/stable/distributed.html#torch.distributed.barrier)
from the PyTorch documentation. Keep in mind these constraints:

- `torch.distributed` is not fully supported on TPU v2/v3 at this time. Only a
  subset of operations with the `xla` backend are tested, and `gloo` will likely
  not work as expected in a multiprocessing context.
- In our experiments, `gloo` does not scale well to thousands of chips, so
  expect this alternative to be less reliable than using `xm.rendezvous` with
  PJRT.

Note: PyTorch/XLA 1.13 implementenation of `xm.rendezvous` uses `gloo` and has
both of the above constraints.

## TPUs v2/v3 vs v4

On TPU v4, one TPU chip is represented to PyTorch as one device, while on TPUs
v2/v3, one TPU chip is represented to PyTorch as _two_ devices. It is not
possible to access the same TPU chip from multiple processes, so workloads must
be able to handle two devices per process. The easiest way to handle this is to
spawn two threads per process on TPU v2/v3, which is done automatically by
`xmp.spawn` when using PjRt. With multiple threads per process, multiple
replicas will share global state, causing the following known issues:

- Threads will share the same `torch` random seed used for parameter
  initialization. If you relied on each process having the same random seed for
  deterministic parameter initialization, you will have to synchronize module
  parameters via collective broadcasting instead (e.g.
  `pjrt.broadcast_master_param(model)`). See [`test_train_mp_imagenet.py`](`../test/test_train_mp_imagenet.py`)
  for an example.
- `torch.distributed` uses a global process group and does not support
  multi-threading, so the `xla` `torch.distributed` backend does not fully
  support TPU v2/v3 with PJRT at this time.

## PjRt and DDP

PjRt composes really well with [the new experimental
torch.nn.parallel.DistributedDataParallel feature](./ddp.md) on TPU V4. Just
run the DDP script as usual but with `PJRT_DEVICE=TPU`. Here is a full example:
```
PJRT_DEVICE=TPU MASTER_ADDR=localhost MASTER_PORT=6000 python xla/test/test_train_mp_mnist.py --ddp --fake_data --num_epochs 1
```

### Experimental PjRt DDP implementation

_New in PyTorch/XLA r1.14 (nightly only)_

Due to `torch.distributed`'s limitations on multithreading,
`torch.nn.parallel.DistributedDataParallel` does not support TPU v2/v3 with
PJRT. Thus, we have provided an alternative implementation of DDP that is
optimized for TPUs and supports TPU v2 and v3 in
[`torch.experimental.pjrt.DistributedDataParallel`](../torch_xla/experimental/pjrt.py).

All of PjRt is in an experimental preview state, but consider this DDP
implementation to be _especially_ unstable. The behavior may change
significantly over time, it may produce incorrect results, or it may be
removed entirely. If you encounter any issues, please report them on GitHub with
the `runtime` and `ddp` tags.

See [`test_train_mp_imagenet.py`](`../test/test_train_mp_imagenet.py`) for an
example drop-in usage.
