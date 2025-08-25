import jax
from jax.sharding import PartitionSpec, NamedSharding
import torch
import torchax
from torchax import interop


def _shard_first_multiple_of(axis_name, shape, multiple_of):
  """Creates a PartitionSpec to shard the first dimension divisible by a number.

  Iterates through the dimensions specified by `shape`. Finds the first dimension
  whose size is a multiple of `multiple_of` and returns a PartitionSpec that
  shards that dimension along the given `axis_name`. All preceding dimensions
  are not sharded (marked as None in the PartitionSpec). All subsequent dimensions
  skipped, which would be implicitly treated as replicated.

  Args:
    axis_name: The name of the mesh axis to shard along (e.g., "data", "mdl").
    shape: A tuple or list representing the shape of the tensor to be sharded.
    multiple_of: The integer value that a dimension size must be divisible by
      in order to be sharded. Typically the size of the mesh axis.

  Returns:
    A jax.sharding.PartitionSpec object specifying how to shard the tensor.
    For example, if shape=(10, 20, 30), axis_name='x', multiple_of=4,
    it would return PartitionSpec(None, 'x', None).
    If none divides then it should return a replicated PartitionSpec
  """
  sharding = []
  found = False
  for size in shape:
    if not found and size % multiple_of == 0:
      found = True
      sharding.append(axis_name)
    else:
      sharding.append(None)
  return PartitionSpec(*sharding)


class SingleAxisSharder:
  """A callable object that generates PartitionSpecs for single-axis sharding.

  This sharder strategy attempts to shard the *first* dimension of a tensor
  that is divisible by the specified `axis_size` along the given `axis_name`.
  It's useful for simple 1D mesh sharding scenarios like FSDP where parameters
  are typically sharded along one dimension.

  Attributes:
    axis_name: The name of the mesh axis to shard along.
    axis_size: The size of the mesh axis (number of devices along that axis).
  """

  def __init__(self, axis_name, axis_size, replicate_unshardable=False):
    """Initializes the SingleAxisSharder.

    Args:
      axis_name: The name of the mesh axis (e.g., "fsdp", "data").
      axis_size: The number of devices along the specified mesh axis.
      replicate_unshardable: indicate whether it should return replicated sharding
        (P()) when none of the axis is divisible by the axis size.
    """
    self.axis_name = axis_name
    self.axis_size = axis_size
    self.replicate_unshardable = replicate_unshardable

  def __call__(self, name, shapedtype):
    """Generates a PartitionSpec for a given tensor name and shaped type.

    Args:
      name: The name of the tensor (e.g., parameter name). This argument is
        provided for compatibility with more complex sharders but is not used
        by this simple sharder.
      shapedtype: An object with a `.shape` attribute describing the tensor's shape,
        and `.dtype` describing it's dtype. Example: jax.Array, jax.ShapeDtypeStruct
        or a torch.Tensor)

    Returns:
      A jax.sharding.PartitionSpec determined by finding the first dimension
      in `shapedtype.shape` divisible by `self.axis_size` using the helper
      `_shard_first_multiple_of`.
    """
    del name
    sharding = _shard_first_multiple_of(self.axis_name, shapedtype.shape,
                                        self.axis_size)
    if not self.replicate_unshardable and all(s is None for s in sharding):
      raise AssertionError(
          f"Unable to find a dim to shard because "
          f"None of the dims ({shapedtype.shape}) in shape is multiple of {self.axis_size}"
      )
    return sharding


class Mesh:
  """A helper class that wraps `jax.sharding.Mesh` object.

  The goal of this class is to provide helper methods that facilitate the
  sharding of PyTorch tensors or models given a JAX device mesh configuration.
  It simplifies initializing models directly into a sharded state.

  Attributes:
    jax_mesh: The underlying `jax.sharding.Mesh` object defining the device grid
      and axis names.
    _sharder: The default sharding strategy callable (like SingleAxisSharder)
      used to determine the PartitionSpec for each parameter if not overridden
      during method calls. Can be None if no default is appropriate or set.
  """

  @classmethod
  def fsdp_mesh(cls, axis_name="fsdp"):
    """Creates a Mesh instance suitable for 1D FSDP-style sharding.

    This named constructor creates a 1D mesh encompassing all available XLA
    devices. It assigns the specified `axis_name` to this single dimension.
    It then creates a `Mesh` instance using this JAX mesh and a
    `SingleAxisSharder` configured appropriately for this 1D mesh.

    Args:
      axis_name: The name to assign to the single mesh axis (default: "fsdp").
        This name will be used by the default `SingleAxisSharder`.

    Returns:
      A Mesh instance configured with a 1D JAX mesh across all devices and a
      corresponding SingleAxisSharder.
    """
    ndevice = jax.device_count()
    jax_mesh = jax.make_mesh((ndevice,), (axis_name,))
    # replicate_unshardable so scalars and small model attributes are replicated.
    return cls(jax_mesh, SingleAxisSharder(axis_name, ndevice, True))

  def __init__(self, jax_mesh, sharder=None):
    """Initializes the Mesh helper.

    Args:
      jax_mesh: A pre-configured `jax.sharding.Mesh` object defining the
        physical device grid and logical axis names.
      sharder: An optional callable (e.g., an instance of SingleAxisSharder)
        that takes (name, shapedtype) and returns a `jax.sharding.PartitionSpec`.
        This serves as the default sharding strategy.
        If None, and the provided `jax_mesh` has exactly one axis, a
        `SingleAxisSharder` is created automatically for that single axis.
        If None and the mesh has multiple axes, `_sharder` remains None, and
        an `override_sharder` must be provided to methods like
        `initialize_model_sharded`.
    """
    self.jax_mesh = jax_mesh
    if sharder is None:
      assert len(self.jax_mesh.axis_names) == 1
      sharder = SingleAxisSharder(self.jax_mesh.axis_names[0],
                                  len(self.mesh.device_ids))
    self._sharder = sharder

  def initialize_model_sharded(self,
                               model_class,
                               init_args,
                               init_kwargs=None,
                               override_sharder=None):
    """Initializes a PyTorch model with its parameters sharded across the mesh.

    This method orchestrates the initialization of a `torch.nn.Module` such
    that its parameters are created directly on the target devices according
    to the sharding specifications derived from the mesh and the chosen sharder.
    It leverages `torchax.interop.jax_jit` to achieve this.

    Args:
      model_class: The PyTorch model class (a subclass of `torch.nn.Module`).
      init_args: A tuple containing the positional arguments required by the
        `model_class.__init__` method.
      init_kwargs: An optional dictionary containing the keyword arguments for
        the `model_class.__init__` method. Defaults to None (treated as {}).
      override_sharder: An optional callable sharding strategy to use
        specifically for this initialization. If provided, it takes precedence
        over the mesh's default `_sharder`. It must accept `(name, shapedtype)`
        and return a `PartitionSpec`. If None, the mesh's default `_sharder`
        is used.

    Returns:
      An instance of `model_class` whose parameters have been initialized and
      are represented by sharded tensors distributed across the devices in the
      `jax_mesh`.

    Raises:
      ValueError: If no sharder is available (i.e., `override_sharder` is None
        and the mesh's default `_sharder` is also None).
      AssertionError: Can be raised by the sharder (e.g., `SingleAxisSharder`)
        if it fails to determine a valid sharding for any parameter.
      TypeError: If `shapedtype` passed to the sharder doesn't have a `.shape`.
      Other errors from JAX JIT compilation or PyTorch model initialization.
    """
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
      with torchax.default_env(), torch.device('meta'):
        model = model_class(*init_args, **init_kwargs)
      return dict(model.state_dict())

    jitted = interop.jax_jit(
        model_initializer, kwargs_for_jax_jit={"out_shardings": output_shards})
    weights_dict = jitted()

    model.load_state_dict(weights_dict, assign=True)
    return model

  def shard_model(self, model, override_sharder=None):
    sharder = override_sharder or self._sharder
    states = model.state_dict()
    output_shards = {
        name: NamedSharding(self.jax_mesh, sharder(name, tensor))
        for name, tensor in states.items()
    }
    model.load_state_dict(output_shards, assign=True)
