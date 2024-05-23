"""Grouped matrix multiplication kernels for TPU written in Pallas."""

from typing import Any, Callable, Optional, Union
from torch_xla.experimental.custom_kernel import jax_import_guard
import torch
import torch_xla
import numpy as np


def gmm(lhs: torch.Tensor, rhs: torch.Tensor,
        group_sizes: torch.Tensor) -> torch.Tensor:
  """Compute lhs[sizes[i-1]:sizes[i], :] @ rhs for each group 'i'.

  Args:
    lhs: A 2d, jnp.ndarray with shape [m, k].
    rhs: A 3d, jnp.ndarray with shape [num_groups, k, n].
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    preferred_element_type: jnp.dtype, the element type for the output matrix.

  Returns:
    A 2d, jnp.ndarray with shape [m, n].
  """
  # Import JAX within the function such that we don't need to call the jax_import_guard()
  # in the global scope which could cause problems for xmp.spawn.
  jax_import_guard()
  import jax
  import jax.numpy as jnp
  from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm, make_group_metadata
  from torch_xla.experimental.custom_kernel import trace_pallas

  payload, _ = trace_pallas(gmm, lhs, rhs, group_sizes)

  m, n = lhs.shape[0], rhs.shape[2]
  # Create the metadata we need for computation.
  group_sizes = jnp.asarray(group_sizes.numpy())
  group_metadata, num_active_tiles = make_group_metadata(
      group_sizes=group_sizes,
      m=lhs.shape[0],
      tm=128,
      start_group=0,
      num_nonzero_groups=rhs.shape[0],
      visit_empty_groups=False,
  )
  group_metadata0 = torch.from_numpy(np.array(group_metadata[0])).to(
      torch.int32).to("xla")
  group_metadata1 = torch.from_numpy(np.array(group_metadata[1])).to("xla")
  group_metadata2 = torch.from_numpy(np.array(group_metadata[2])).to("xla")
  num_active_tiles = torch.tensor(np.array(num_active_tiles)).to("xla")
  group_offset_torch = torch.tensor([0], dtype=torch.int32).to("xla")
  output_shape = torch.Size([m, n])
  out = torch_xla._XLAC._xla_tpu_custom_call([
      num_active_tiles, group_metadata0, group_metadata1, group_metadata2,
      group_offset_torch, lhs, rhs
  ], payload, [output_shape], [lhs.dtype])
  return out
