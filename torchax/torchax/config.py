import dataclasses


@dataclasses.dataclass
class Configuration:
  """A dataclass for configuring the behavior of `torchax`.

  **Attributes:**

  *   `debug_print_each_op` (`bool`): If `True`, prints each operation as it is
      dispatched.
  *   `debug_accuracy_for_each_op` (`bool`): If `True`, checks the accuracy of
      each operation by comparing its output with the equivalent PyTorch
      operation on the CPU.
  *   `debug_mixed_tensor` (`bool`): If `True`, enables debugging for mixed
      tensor operations.
  *   `debug_print_each_op_operands` (`bool`): If `True`, prints the operands of
      each operation.
  *   `use_int32_for_index` (`bool`): If `True`, uses `int32` for indexing
      operations.
  *   `allow_mixed_math_with_scalar_tensor` (`bool`): If `True`, allows mixed
      math operations between `torchax.Tensor` and scalar `torch.Tensor`s.
  *   `force_materialize_views` (`bool`): If `True`, eagerly materializes `View`
      objects into `torchax.Tensor`s.
  *   `use_dlpack_for_data_conversion` (`bool`): If `True`, uses DLPack for
      converting between `jax.Array` and `torch.Tensor`.
  *   `use_tpu_flash_attention` (`bool`): If `True`, uses TPU-optimized flash
      attention.
  *   `shmap_flash_attention` (`bool`): If `True`, uses `shard_map` for flash
      attention.
  *   `treat_cuda_as_jax_device` (`bool`): If `True`, treats CUDA devices as JAX
      devices.
  *   `internal_respect_torch_return_dtypes` (`bool`): If `True`, respects the
      return data types of PyTorch operations.
  """
  debug_print_each_op: bool = False
  debug_accuracy_for_each_op: bool = False
  debug_mixed_tensor: bool = False
  debug_print_each_op_operands: bool = False

  use_int32_for_index: bool = False

  # normally, math between CPU torch.Tensor with torchax.Tensor is not
  # allowed. However, if that torch.Tensor happens to be scalar, then we
  # can use scalar * tensor math to handle it
  allow_mixed_math_with_scalar_tensor: bool = True

  # If true, we will convert Views into torchax.Tensors eagerly
  force_materialize_views: bool = False

  # Use DLPack for converting jax.Arrays <-> and torch.Tensor
  use_dlpack_for_data_conversion: bool = False

  # Flash attention
  use_tpu_flash_attention: bool = False
  shmap_flash_attention: bool = False

  # device
  treat_cuda_as_jax_device: bool = True
  internal_respect_torch_return_dtypes: bool = False
