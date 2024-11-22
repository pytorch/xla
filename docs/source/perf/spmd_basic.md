# PyTorch/XLA SPMD User Guide

In this user guide, we discuss how
[GSPMD](https://arxiv.org/abs/2105.04663) is integrated in PyTorch/XLA,
and provide a design overview to illustrate how the SPMD sharding
annotation API and its constructs work.

## What is PyTorch/XLA SPMD?

[GSPMD](https://arxiv.org/abs/2105.04663) is an automatic
parallelization system for common ML workloads. The XLA compiler will
transform the single device program into a partitioned one with proper
collectives, based on the user provided sharding hints. This feature
allows developers to write PyTorch programs as if they are on a single
large device without any custom sharded computation ops and/or
collective communications to scale.

![Execution strategies](../_static/img/spmd_mode.png "image_tooltip")
_<span style="text-decoration:underline;">Figure 1. Comparison of two different execution strategies, (a) for non-SPMD and (b) for SPMD.</span>_

## How to use PyTorch/XLA SPMD?

Here is an simple example of using SPMD

``` python
import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh


# Enable XLA SPMD execution mode.
xr.use_spmd()


# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()
mesh_shape = (num_devices, 1)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('data', 'model'))


t = torch.randn(8, 4).to(xm.xla_device())


# Mesh partitioning, each device holds 1/8-th of the input
partition_spec = ('data', 'model')
xs.mark_sharding(t, mesh, partition_spec)
```

Let's explain these concepts one by one

### SPMD Mode

In order to use SPMD, you need to enable it via `xr.use_spmd()`. In SPMD
mode there is only one logical device. Distributed computation and
collective is handled by the `mark_sharding`. Note that user can not mix
SPMD with other distributed libraries.

### Mesh

For a given cluster of devices, a physical mesh is a representation of
the interconnect topology.

1.  `mesh_shape` is a tuple that will be multiplied to the total number
    of physical devices.
2.  `device_ids` is almost always `np.array(range(num_devices))`.
3.  Users are also encouraged to give each mesh dimension a name. In the
    above example, the first mesh dimension is the `data` dimension and
    the second mesh dimension is the `model` dimension.

You can also check more mesh info via

``` python
    >>> mesh.shape()
    OrderedDict([('data', 4), ('model', 1)])
```

### Partition Spec

partition_spec has the same rank as the input tensor. Each dimension
describes how the corresponding input tensor dimension is sharded across
the device mesh. In the above example tensor `t`'s fist dimension is
being sharded at `data` dimension and the second dimension is being
sharded at `model` dimension.

User can also shard tensor that has different dimensions from the mesh
shape.

``` python
t1 = torch.randn(8, 8, 16).to(device)
t2 = torch.randn(8).to(device)

# First dimension is being replicated.
xs.mark_sharding(t1, mesh, (None, 'data', 'model'))

# First dimension is being sharded at data dimension. 
# model dimension is used for replication when omitted.
xs.mark_sharding(t2, mesh, ('data',))

# First dimension is sharded across both mesh axes.
xs.mark_sharding( t2, mesh, (('data', 'model'),))
```

## Further Reading

1.  [Example](https://github.com/pytorch/xla/blob/master/examples/data_parallel/train_resnet_spmd_data_parallel.py)
    to use SPMD to express data parallism.
2.  [Example](https://github.com/pytorch/xla/blob/master/examples/fsdp/train_decoder_only_fsdp_v2.py)
    to use SPMD to express FSDP(Fully Sharded Data Parallel).
3.  [SPMD advanced
    topics](https://github.com/pytorch/xla/blob/master/docs/spmd_advanced.rst)
4.  [Spmd Distributed
    Checkpoint](https://github.com/pytorch/xla/blob/master/docs/spmd_distributed_checkpoint.rst)
