import dataclasses


@dataclasses.dataclass
class Configuration:
    debug_accuracy_for_each_op: bool = False
    use_int32_for_index: bool = False

    # Flash attention
    use_tpu_flash_attention: bool = False
    shmap_flash_attention: bool = False