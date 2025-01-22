import dataclasses


@dataclasses.dataclass
class Configuration:
    debug_print_each_op: bool = False
    debug_accuracy_for_each_op: bool = False
    debug_mixed_tensor: bool = False
    use_int32_for_index: bool = False

    # Flash attention
    use_tpu_flash_attention: bool = False
    shmap_flash_attention: bool = False

    # device
    treat_cuda_as_jax_device: bool = True
    use_torch_native_for_cpu_tensor: bool = True
    internal_respect_torch_return_dtypes: bool = False
