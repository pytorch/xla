import jax
from jax.sharding import PartitionSpec, NamedSharding
import torch
import torchax
from torchax import interop


def _shard_first_multiple_of(axis_name, shape, multiple_of):
  sharding = []
  for size in shape:
    if size % multiple_of == 0:
      break
  sharding.append(axis_name)
  return PartitionSpec(*sharding)


class SingleAxisSharder:
  def __init__(self, axis_name, axis_size):
    self.axis_name = axis_name
    self.axis_size = axis_size

  def __call__(self, name, shapedtype):
    return _shard_first_multiple_of(
      self.axis_name, shapedtype.shape, self.axis_size
    )


class Mesh:
  @classmethod
  def fsdp_mesh(cls, axis_name="fsdp"):
    ndevice = jax.device_count()
    jax_mesh = jax.make_mesh((ndevice,), (axis_name,))
    return cls(jax_mesh, SingleAxisSharder(axis_name, ndevice))

  def __init__(self, jax_mesh, sharder=None):
    self.jax_mesh = jax_mesh
    if sharder is None:
      assert len(self.mesh.axis_names) == 1
      sharder = SingleAxisSharder(
        self.mesh.axis_names[0], len(self.mesh.device_ids)
      )
    self._sharder = sharder

  def initialize_model_sharded(
    self, model_class, init_args, init_kwargs=None, override_sharder=None
  ):
    init_kwargs = init_kwargs or {}
    with torch.device("meta"), torchax.disable_temporarily():
      model = model_class(*init_args, **init_kwargs)

    sharder = override_sharder or self._sharder

    states = model.state_dict()
    output_shards = {
      name: NamedSharding(self.jax_mesh, sharder(name, tensor))
      for name, tensor in states.items()
    }

    def model_initializer():
      model = model_class(*init_args, **init_kwargs)
      return dict(model.state_dict())

    jitted = interop.jax_jit(
      model_initializer, kwargs_for_jax_jit={"out_shardings": output_shards}
    )
    weights_dict = jitted()

    model.load_state_dict(weights_dict, assign=True)
    return model
