from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)


# TODO Convert this to a DTensorSpec subclass
@dataclass
class ShardingSpec(DTensorSpec):
  mesh: Mesh
  partition_spec: Tuple[Union[int, None]]

  def apply(self, t: torch.Tensor):
    assert (t.device == xm.xla_device())
    assert (xu.check_env_flag('XLA_USE_SPMD'))
    mark_sharding(t, self.mesh, self.partition_spec)


# TODO should this be a torch tensor or XLA tensor?
class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: 'XLAShard'):
        ctx.dtensor_spec = input._spec
        return input._local_tensor.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dtensor_spec = ctx.dtensor_spec
        dtensor_meta = dtensor_spec.tensor_meta

        # TODO Initialize XLA Shard here
        return DTensor(
            grad_output,
            dtensor_spec.mesh,
            dtensor_spec.placements,
            shape=dtensor_meta.shape,
            dtype=dtensor_meta.dtype,
            requires_grad=grad_output.requires_grad,
            stride=dtensor_meta.stride,
        )


class XLAShard(DTensor):
    _local_tensor: torch.Tensor
    _spec: ShardingSpec

    def to_local(self) -> torch.Tensor:
        return _ToTorchTensor.apply(self)
