def all_aten_jax_ops():
    # to load the ops
    import torch_xla2.jaten # type: ignore
    import torch_xla2.ops_registry # type: ignore
    return {
        key: val.func 
        for key, val in torch_xla2.ops_registry.all_aten_ops
        if val.is_jax_function
    }