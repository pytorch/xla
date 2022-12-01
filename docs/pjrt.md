# Experimental PjRt Runtime Support

The PyTorch/XLA team is currently migrating from the currently-supported XRT
runtime to the [PjRt
runtime](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/pjrt)
used by [JAX](https://github.com/google/jax). Although PjRt may work on TPU v2
and v3, we plan on making PjRt the officially supported runtime for PyTorch/XLA
on TPU v4 and future generations of TPU.

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

### v4 TPU

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

Note: `xmp.spawn`'s `nprocs` argument is not implemented for PjRt.

#### Pods

On TPU Pods, use `gcloud` to run your command on each TPU in parallel:

```
gcloud alpha compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command="git clone --depth=1 --branch r1.13 https://github.com/pytorch/xla.git"
gcloud alpha compute tpus tpu-vm ssh $USER-pjrt --zone=us-central2-b --project=$PROJECT --worker=all --command="PJRT_DEVICE=TPU python3 xla/test/test_train_mp_imagenet.py --fake_data --batch_size=256 --num_epochs=1"
```

### GPU

Coming soon in a future release!

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

## TPUs v2/v3 vs v4

On TPU v4, one TPU chip is represented to PyTorch as one device, while on TPUs
v2/v3, one TPU chip is represented to PyTorch as _two_ devices. It is not
possible to access the same TPU chip from multiple processes, so workloads must
be able to handle two devices per process. The easiest way to handle this is to
spawn two threads per process on TPU v2/v3, which is done automatically by
`xmp.spawn` when using PjRt. With multiple threads per process, multiple replicas
will share global state, causing the following known issues:

- Threads will share the same `torch` random seed used for parameter
  initialization. If you relied on each process having the same random seed for
  deterministic parameter initialization, you will have to synchronize module
  parameters via collective broadcasting instead (e.g.
  `pjrt.broadcast_master_param(model)`).
- `torch.distributed` uses a global process group and does not support
  multi-threading, so the `xla` `torch.distributed` backend will not work with
  PjRt and TPU v2 and v3 at this time.
- Because the current implementation of `xm.rendezvous` for PjRt relies on
  `torch.distributed`, `xm.rendezvous` is not supported with PjRt on TPU v2 and
  v3.

### Compatible examples

For an overview of the changes required to migrate from TPU v2/v3 to v4, compare
our MNIST ([XRT](../test/test_train_mp_mnist.py),
[PjRt](../test/pjrt/test_train_pjrt_mnist.py)) and ImageNet
([XRT](../test/test_train_mp_imagenet.py),
[PjRt](../test/pjrt/test_train_pjrt_imagenet.py)) examples.

The PjRt MNIST and ImageNet examples are compatible with all versions of TPU.
Use the following commands to run them on a single-host TPU (e.g. v3-8 or v4-8).

```
PJRT_DEVICE=TPU python3 xla/test/pjrt/test_train_pjrt_mnist.py --fake_data
PJRT_DEVICE=TPU python3 xla/test/pjrt/test_train_pjrt_imagenet.py --fake_data
```

## PjRt and DDP

PjRt composes really well with [the new experimental
torch.nn.parallel.DistributedDataParallel feature](./ddp.md) on TPU V4. Just
run the DDP script as usual but with `PJRT_DEVICE=TPU`. Here is a full example:
```
PJRT_DEVICE=TPU MASTER_ADDR=localhost MASTER_PORT=6000 python xla/test/test_train_mp_mnist.py --ddp --fake_data --num_epochs 1
```

Caveat: for TPU V2 and V3, however, XRT will still be needed to run DDP.
