import dataclasses


@dataclasses.dataclass
class Configuration:
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
