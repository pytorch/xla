from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch_xla


def create_global_tensor_from_shards(shape: Tuple[int],
                                     sharding: torch_xla._XLAC.OpSharding,
                                     shards: Sequence[torch.Tensor]):
  """
  Similar to jax.make_array_from_single_device_arrays
  """
  # Now this function relies on caller to pass matching sharding and shape.
  # TODO(lsy323): Check if shape, sharding and shards are matching.
  if shards[0].device.type == 'cpu':
    assert all(
        s.device.type == 'cpu' for s in shards), "All shards must be on CPU."
    from_cpu_shards = torch_xla._XLAC._global_tensor_from_tpu_shards
    return from_cpu_shards(shards, sharding, shape)
  else:
    from_tpu_shards = torch_xla._XLAC._global_tensor_from_tpu_shards
    return from_tpu_shards(shards, sharding, shape)
