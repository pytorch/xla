
# PyTorch/XLA SPMD User Guide

In this user guide, we discuss how [GSPMD](https://arxiv.org/abs/2105.04663) is integrated in PyTorch/XLA, and provide a design overview to illustrate how the SPMD sharding annotation API and its constructs work. And then, we provide a list of reference examples for users to try.


## What is PyTorch/XLA SPMD?

[GSPMD](https://arxiv.org/abs/2105.04663) is an automatic parallelization system for common ML workloads. The XLA compiler will transform the single device program into a partitioned one with proper collectives, based on the user provided sharding hints. This feature allows developers to write PyTorch programs as if they are on a single large device without any custom sharded computation ops and/or collective communications to scale.

![alt_text](assets/spmd_mode.png "image_tooltip")
_<span style="text-decoration:underline;">Figure 1. Comparison of two different execution strategies, (a) for non-SPMD and (b) for SPMD.</span>_

To support GSPMD in PyTorch/XLA, we are introducing a new execution mode. Before GSPMD, the execution mode in PyTorch/XLA assumed multiple model replicas, each with a single core (Figure 1.a). This mode of execution, as illustrated in the above  suits data parallelism frameworks, like the popular PyTorch [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) or Fully Sharded Data Parallel (FSDP), but is also limited in that a replica can only reside on one device core for execution. PyTorch/XLA SPMD introduces a new execution mode that assumes a single replica with multiple cores (Figure 1.b), allowing a replica to run across multiple device cores. This shift unlocks more advanced parallelism strategies for better large model training performance.

PyTorch/XLA SPMD is available on the new [PJRT](https://github.com/pytorch/xla/blob/master/docs/pjrt.md) runtime. To enable PyTorch/XLA SPMD execution mode, the user must call <code>[use_spmd() API](https://github.com/pytorch/xla/blob/b8b484515a97f74e013dcf38125c44d53a41f011/torch_xla/runtime.py#L214)</code>.

```python
import torch_xla.runtime as xr

# Enable PyTorch/XLA SPMD execution mode.
xr.use_spmd()
assert xr.is_spmd() == True
```

It is important to note that SPMD is a replacement for any existing parallel mechanisms, including DDP and FSDP. Users can not mix two different execution modes (SPMD and non-SPMD), and later in this guide we will go over how to use SPMD annotation to perform DDP and FSDP.

Also, this version of the SPMD is currently only tested.optimized on Google Cloud TPU. GPU support and optimization will come in the 2.2 release.


## PyTorch/XLA SPMD Design Overview


### Simple Example & Sharding Aannotation API

Users can annotate native PyTorch tensors using the `mark_sharding` API ([src](https://github.com/pytorch/xla/blob/9a5fdf3920c18275cf7dba785193636f1b39ced9/torch_xla/experimental/xla_sharding.py#L388)). This takes `torch.Tensor` as input and returns a `XLAShardedTensor` as output.

```python
def mark_sharding(t: Union[torch.Tensor, XLAShardedTensor], mesh: Mesh, partition_spec: Tuple[Union[int, None]]) -> XLAShardedTensor
```

Invoking `mark_sharding` API takes a user defined logical [mesh](#mesh) and [partition\_spec](#partition-spec) and generates a sharding annotation for the XLA compiler. The sharding spec is attached to the XLATensor. Here is a simple usage example from the [[RFC](https://github.com/pytorch/xla/issues/3871), to illustrate how the sharding annotation API works:

```python
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharding import Mesh

# Enable XLA SPMD execution mode.
xr.use_spmd()

# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
mesh_shape = (2, 4)
num_devices = xr.global_runtime_device_count()
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

t = torch.randn(8, 4).to(xm.xla_device())

# Mesh partitioning, each device holds 1/8-th of the input
partition_spec = (0, 1)
m1_sharded = xs.mark_sharding(t, mesh, partition_spec)
assert isinstance(m1_sharded, XLAShardedTensor) == True
```

We can annotate different tensors in the PyTorch program to enable different parallelism techniques, as described in the comment below:

```python
# Sharding annotate the linear layer weights.
model = SimpleLinear().to(xm.xla_device())
xs.mark_sharding(model.fc1.weight, mesh, partition_spec)

# Training loop
model.train()
for step, (data, target) in enumerate(loader):
  # Assumes `loader` returns data, target on XLA device
  optimizer.zero_grad()
  # Sharding annotate input data, we can shard any input
  # dimensions. Sharidng the batch dimension enables
  # in data parallelism, sharding the feature dimension enables
  # spatial partitioning.
  xs.mark_sharding(data, mesh, partition_spec)
  ouput = model(data)
  loss = loss_fn(output, target)
  optimizer.step()
  xm.mark_step()
```

More complete unit test cases and integration test examples are available in the PyTorch/XLA [repo](https://github.com/pytorch/xla/tree/r2.0/test/spmd).


### Mesh

For a given cluster of devices, a physical mesh is a representation of the interconnect topology.

We derive a logical mesh based on this topology to create sub-groups of devices which can be used for partitioning different axes of tensors in a model.

![alt_text](assets/mesh_spmd2.png "image_tooltip")

We abstract logical mesh with [Mesh API](https://github.com/pytorch/xla/blob/028df4da388468fa9a41b1f98ea08bfce13b4c63/torch_xla/experimental/xla_sharding.py#L16). The axes of the logical Mesh can be named. Here is an example:

```python
import torch_xla.runtime as xr
from torch_xla.experimental.xla_sharding import Mesh

# Assuming you are running on a TPU host that has 8 devices attached
num_devices = xr.global_runtime_device_count()
# mesh shape will be (4,2) in this example
mesh_shape = (num_devices // 2, 2)
device_ids = np.array(range(num_devices))
# axis_names 'x' nad 'y' are optional
mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))

mesh.get_logical_mesh()
>> array([[0, 1],
          [2, 3],
          [4, 5],
          [6, 7]])
mesh.shape()
>> OrderedDict([('x', 4), ('y', 2)])
```

In general, SPMD programs should create a single mesh and reuse it for all sharding to ensure that the tiling assignment is consistent with the intended sharding strategy. The same mesh can be reused for tensors of different shapes and shardings by manipulating the partition spec, described further below.

### Hybrid Mesh

Mesh nicely abstracts how the physical device mesh is constructed. Users can arrange devices in any shape and order using the logical mesh. However, one can define a more performant mesh based on the physical topology, especially when it involves Data Center Network (DCN) cross slice connections. HybridMesh creates a mesh which gives good performance out of the box for such multislice environments. It accepts ici\_mesh\_shape and dcn\_mesh\_shape which denote logical mesh shapes of inner and outer network.

```python
from torch_xla.experimental.xla_sharding import HybridMesh

# This example is assuming 2 slices of v4-8.
# - ici_mesh_shape: shape of the logical mesh for inner connected devices.
# - dcn_mesh_shape: shape of logical mesh for outer connected devices.
ici_mesh_shape = (1, 4, 1) # (data, fsdp, tensor)
dcn_mesh_shape = (2, 1, 1)

mesh = HybridMesh(ici_mesh_shape, dcn_mesh_shape, ('data','fsdp','tensor'))
print(mesh.shape())
>> OrderedDict([('data', 2), ('fsdp', 4), ('tensor', 1)])
```


### Partition Spec

partition\_spec has the same rank as the input tensor. Each dimension describes how the corresponding input tensor dimension is sharded across the device mesh (logically defined by mesh\_shape). `partition_spec` is a tuple of `device_mesh` dimension `index` or None. The index can be an `int` or `str`, if the corresponding mesh dimension is named. This specifies how each input rank is sharded (`index` to `mesh_shape`) or replicated (`None`).

```python
# Provide optional mesh axis names and use them in the partition spec
mesh = Mesh(device_ids, (4, 2), ('data', 'model'))
partition_spec = ('model', 'data')
xs.mark_sharding(input_tensor, mesh, partition_spec)
```

We support all three types of sharding, described in the original [GSPMD](https://arxiv.org/abs/2105.04663) paper. For instance, one can specify partial replication like this:

```python
# Provide optional mesh axis names and use them in the partition spec
mesh = Mesh(device_ids, (2, 2, 2), ('x', 'y', 'z'))

# evenly shard across x and z and replicate among y
partition_spec = ('x', 'z')  # equivalent to ('x', None, 'z')
xs.mark_sharding(input_tensor, mesh, partition_spec)
```

The partition spec enables reuse of the same mesh for different tensor shapes and desired sharding strategies. The following example demonstrates this using a 3D mesh:

```python
# Create a 3-D mesh of 8 devices with logical dimensions replica, fsdp, and
# tensor
mesh = Mesh(device_ids, (2, 2, 2), ('replica', 'fsdp', 'tensor'))

# A 2D tensor can be sharded along the fsdp and tensor axes and replicated
# along the replica axis by omitting `replica` from the partition spec.
two_d_partially_replicated = torch.randn(64, 64, device='xla')
xs.mark_sharding(two_d_partially_replicated, mesh, ('fsdp', 'tensor'))

# A 2D tensor can be sharded across all dimensions by combining, for example,
# the replica and fsdp mesh axes using a tuple
two_d_fully_sharded = torch.randn(64, 64, device='xla')
xs.mark_sharding(two_d_fully_sharded, mesh, (('replica', 'fsdp'), 'tensor'))

# A 4D tensor can be sharded along up to three of its axes using the 3D mesh
four_d = torch.randn(64, 64, 64, 64, device='xla')
xs.mark_sharding(four_d, ('replica', 'fsdp', None, 'tensor'))
```


### XLAShardedTensor

The main use case for `XLAShardedTensor` [[RFC](https://github.com/pytorch/xla/issues/3871)] is to annotate a native `torch.tensor` (on a single device) with a sharding spec. The annotation takes place immediately, but the actual sharding of the tensor is delayed as the computation is carried out lazily, except for the input tensors which are sharded without delay. Once a tensor is annotated and wrapped inside a `XLAShardedTensor`, it can be passed to existing PyTorch ops and `nn.Module` layers as `torch.Tensor`. This is important to ensure that the same PyTorch layers and tensor ops can be stacked together with `XLAShardedTensor`. This means that the user does not need to rewrite the existing ops and model codes for sharded computation. Namely, `XLAShardedTensor` will satisfy the following requirements:



*   `XLAShardedTensor` is a `torch.Tensor` subclass and works directly with native torch ops and `module.layers`. We use `__torch_dispatch__` to send `XLAShardedTensor` to the XLA backend. PyTorch/XLA retrieves attached sharding annotations to trace the graph and invokes XLA SPMDPartitioner.
*   Internally, `XLAShardedTensor` (and its global\_tensor input) is backed by `XLATensor` with a special data structure holding references to the sharded device data.
*   The sharded tensor after lazy execution may be gathered and materialized back to the host as global\_tensor when requested on the host (e.g., printing the value of the global tensor.
*   The handles to the local shards are materialized strictly after the lazy execution. `XLAShardedTensor` exposes [local\_shards](https://github.com/pytorch/xla/blob/909f28fa4c1a44efcd21051557b3bcf2d399620d/torch_xla/experimental/xla_sharded_tensor.py#L111) to return the local shards on addressable devices as <code>List[[XLAShard](https://github.com/pytorch/xla/blob/909f28fa4c1a44efcd21051557b3bcf2d399620d/torch_xla/experimental/xla_sharded_tensor.py#L12)]</code>.

There is also an ongoing effort to integrate <code>XLAShardedTensor</code> into <code>DistributedTensor</code> API to support XLA backend [[RFC](https://github.com/pytorch/pytorch/issues/92909)].

### DTensor Integration
PyTorch has prototype-released [DTensor](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/README.md) in 2.1.
We are integrating PyTorch/XLA SPMD into DTensor API [RFC](https://github.com/pytorch/pytorch/issues/92909). We have a proof-of-concept integration for `distribute_tensor`, which calls `mark_sharding` annotation API to shard a tensor and its computation using XLA:
```python
import torch
from torch.distributed import DeviceMesh, Shard, distribute_tensor

# distribute_tensor now works with `xla` backend using PyTorch/XLA SPMD.
mesh = DeviceMesh("xla", list(range(world_size)))
big_tensor = torch.randn(100000, 88)
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(0)])
```

This feature is experimental and stay tuned for more updates, examples and tutorials in the upcoming releases.

### Sharding-Aware Host-to-Device Data Loading

PyTorch/XLA SPMD takes a single-device program, shards and executes it in parallel. The SPMD execution requires using the native PyTorch DataLoader, which transfers data synchronously from the host to XLA devices. This blocks the training during the input data transfer every step. To improve the native data loading performance, we made PyTorch/XLA ParallelLoader support input sharding directly (src), when passed the optional kwarg _input\_sharding_:

```python
# MpDeviceLoader returns ParallelLoader.per_device_loader as iterator
train_loader = pl.MpDeviceLoader(
         train_loader,  # wraps PyTorch DataLoader
         device,
	  # optional input_sharding field
         input_sharding=xs.ShardingSpec(input_mesh, (0, 1, 2, 3)))
```


### Distributed Checkpointing

PyTorch/XLA SPMD is compatible with the [torch.distributed.checkpoint](https://pytorch.org/docs/stable/distributed.checkpoint.html) library through a dedicated `Planner` instance. Users are able to synchronously save and load checkpoints through this common interface.

The SPMDSavePlanner and SPMDLoadPlanner ([src](https://github.com/pytorch/xla/blob/master/torch_xla/experimental/distributed_checkpoint.py)) classes enable the `save_state_dict` and `load_state_dict` functions to operate directly on the shards of an `XLAShardedTensor`, enabling all of the benefits of distributed checkpointing in SPMD training.

Here is a demonstration of the synchronous distributed checkpointing API:

```python
import torch.distributed.checkpoint as dist_cp
import torch_xla.experimental.distributed_checkpoint as xc

# Saving a state_dict
state_dict = {
    "model": model.state_dict(),
    "optim": optim.state_dict(),
}

dist_cp.save_state_dict(
    state_dict=state_dict,
    storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
    planner=xc.SPMDSavePlanner(),
)
...

# Loading the model's state_dict from the checkpoint. The model should
# already be on the XLA device and have the desired sharding applied.
state_dict = {
    "model": model.state_dict(),
}

dist_cp.load_state_dict(
    state_dict=state_dict,
    storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
    planner=xc.SPMDLoadPlanner(),
)
model.load_state_dict(state_dict["model"])
```

### Virtual Device Optimization

PyTorch/XLA normally transfers tensor data asynchronously from host to device once the tensor is defined. This is to overlap the data transfer with the graph tracing time. However, because GSPMD allows the user to modify the tensor sharding _after _the tensor has been defined, we need an optimization to prevent unnecessary transfer of tensor data back and forth between host and device. We introduce Virtual Device Optimization, a technique to place the tensor data on a virtual device SPMD:0 first, before uploading to the physical devices when all the sharding decisions are finalized. Every tensor data in SPMD mode is placed on a virtual device, SPMD:0. The virtual device is exposed to the user as an XLA device XLA:0 with the actual shards on physical devices, like TPU:0, TPU:1, etc.


### Number of processes

Unlike existing DDP and FSDP, under the SPMD mode, there is always a single process running on each accelerator host. This provides the benefit that PyTorch/XLA only need to compile each graph once which can be reused for all accelerators attached to this host.


### Running SPMD on TPU Pod

There is no code change required to go from single TPU host to TPU Pod if you construct your mesh and partition spec based on the number of devices instead of some hardcode constant. To run the PyTorch/XLA workload on TPU Pod, please refer to the [Pods section](https://github.com/pytorch/xla/blob/master/docs/pjrt.md#pods) of our PJRT guide.


## Reference Examples


### Use SPMD to express Data Parallel

The SPMD API is general enough to express both data parallelism and model parallelism.  One can implement data parallelism simply by annotating the input batch dimension for sharding. Here, we have shard the batch dimension across all available devices (N-way):There are 2 ways of using SPMD to express data parallel or batch sharding:

```python
num_devices = xr.global_runtime_device_count()

# Assume data is 4d and 0th dimension is the batch dimension
mesh_shape = (num_devices, 1, 1, 1)
input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
partition_spec = range(num_devices)

# Shard the batch dimension
xs.mark_sharding(input_tensor, input_mesh, partition_spec)
```

PyTorch/XLA’s MpDeviceLoader supports input batch sharding, which also loads the batches to the devices in the background:

```python
num_devices = xr.global_runtime_device_count()

# Assume data is 4d and 0th dimension is the batch dimension
mesh_shape = (num_devices, 1, 1, 1)
input_mesh = xs.Mesh(device_ids, mesh_shape, ('B', 'C', 'W', 'H'))
partition_spec = range(num_devices)

# Use MpDeviceLoader to load data in background
train_loader = pl.MpDeviceLoader(
     train_loader,
     device,
     input_sharding=xs.ShardingSpec(input_mesh, partition_spec))
```

We highly recommend the second approach as it should yield a better training performance.


### Use SPMD to express FSDP(Fully Sharded Data Parallel)

PyTorch’s FSDP is data parallel + sharded model parameters at 0th dimension. Users first need to use SPMD to express Data Parallels as suggested in the previous section.

```python
for name, param in model.named_parameters():
    shape = (num_devices,) + (1,) * (len(param.shape) - 1)
    mesh = xs.Mesh(device_ids, shape)
    xs.mark_sharding(param, mesh, range(len(param.shape)))
```


### Running Resnet50 example with SPMD

We provided a quick example of [resnet50](https://github.com/pytorch/xla/blob/master/test/spmd/test_train_spmd_imagenet.py) with a couple different SPMD sharding strategies for you to play around with. You can first run it without SPMD using

```bash
python test/spmd/test_train_spmd_imagenet.py --fake_data --batch_size 512
```

and check the throughput. After that you can enable the batch sharding with

```bash
XLA_USE_SPMD=1 python test/spmd/test_train_spmd_imagenet.py --fake_data --batch_size 2048 --model=resnet50 --sharding=batch
```

Note that I used a batch size 4 times as large since I am running it on a TPU v4 which has 4 TPU devices attached to it. You should see the throughput becomes roughly 4x the non-spmd run.
