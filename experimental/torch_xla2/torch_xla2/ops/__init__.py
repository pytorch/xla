def all_aten_jax_ops():
  # to load the ops
  import torch_xla2.ops.jaten  # type: ignore
  import torch_xla2.ops.ops_registry  # type: ignore

  return {
      key: val.func
      for key, val in torch_xla2.ops.ops_registry.all_aten_ops.items()
      if val.is_jax_function
  }
