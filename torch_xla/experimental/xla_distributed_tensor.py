from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
)

# TODO convert this to a device mesh subclass
class Mesh(DeviceMesh):
  device_ids: np.ndarray
  mesh_shape: Tuple[int, ...]
  axis_names: Tuple[str, ...]

  def __init__(self,
               device_ids: Union[np.ndarray, List],
               mesh_shape: Tuple[int, ...],
               axis_names: Tuple[str, ...] = None):
    if not isinstance(device_ids, np.ndarray):
      device_ids = np.array(device_ids)
    assert (axis_names is None) or (len(mesh_shape) == len(axis_names))
    assert (len(device_ids) == np.prod(mesh_shape))
    assert len(device_ids) == len(np.unique(device_ids))
    self.device_ids = device_ids
    self.mesh_shape = mesh_shape
    self.axis_names = axis_names
    assert all(d < self.size() for d in device_ids)

  def size(self):
    return np.prod(self.mesh_shape)

  def shape(self):
    return OrderedDict(
        (name, size) for name, size in zip(self.axis_name, self.mesh_shape))

  def get_logical_mesh(self):
    return self.device_ids.reshape(self.mesh_shape)


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

# TODO convert to dtensor subclass
class XLAShard(DTensor):
    _local_tensor: torch.Tensor
    _spec: ShardingSpec

    def to_local(self, rank: int = 0) -> torch.Tensor:
        return _ToTorchTensor.apply(self)

    @property
    def device_mesh(self) -> DeviceMesh:
        return self._spec.mesh

    @property
    def placements(self) -> Sequence[Placement]:
        return self._spec.placements

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
    ) -> "DTensor":
        NotImplementedError

    @classmethod
    def from_local(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        run_check: bool = True,
    ) -> "DTensor":
        NotImplementedError