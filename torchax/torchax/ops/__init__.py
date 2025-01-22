def all_aten_jax_ops():
  # to load the ops
  import torchax.ops.jaten  # type: ignore
  import torchax.ops.ops_registry  # type: ignore

  return {
      key: val.func
      for key, val in torchax.ops.ops_registry.all_aten_ops.items()
      if val.is_jax_function
  }
