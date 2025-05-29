"""An example of how PyTorch native tensor parallelism could work with XLA.
Currently fails because `redistribute` is not defined for XLAShardedTensor."""
import sys
import os
example_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(example_folder)
from train_decoder_only_base import TrainDecoderOnlyBase

from torch.distributed.tensor import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
import torch_xla.runtime as xr

class TrainDecoderSpmdTP(TrainDecoderOnlyBase):

  def __init__(self):
    super().__init__()
    tp_mesh = init_device_mesh("xla", (xr.global_runtime_device_count(),))
    layer_tp_plan = {
        # by default ColwiseParallel input layouts is replicated
        # and RowwiseParallel output layouts is replicated
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
    }

    for layer in self.model.layers:
      parallelize_module(
        module=layer,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan
      )

if __name__ == '__main__':
  xr.use_spmd()
  TrainDecoderSpmdTP().start_training()