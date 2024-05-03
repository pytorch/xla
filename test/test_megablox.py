"""Grouped matrix multiplication kernels for TPU written in Pallas."""

from typing import Optional, Union, Callable
import torch
import torch_xla
import numpy as np
import torch_xla.experimental.megablox.gmm as g


def jax_import_guard():
  # Somehow, we need to grab the TPU before JAX locks it. Otherwise, any pt-xla TPU operations will hang.
  torch_xla._XLAC._init_computation_client()


def _reference_gmm(
    lhs: np.array,
    rhs: np.array,
    group_sizes: np.array,
    preferred_element_type: np.dtype = np.float32,
) -> np.array:

  start = 0
  out = []
  for i, size in enumerate(group_sizes):
    result = np.dot(lhs[start:start + size, :], rhs[i, :, :])

    result = result.astype(preferred_element_type)
    out.append(result)
    start += group_sizes[i]
  return np.array(np.concatenate(out, axis=0))


def _group_sizes_strategy(m: int, num_groups: int) -> torch.Tensor:
  # Randomly sample the ends of the groups in the m-dimension. Let the fuzzer
  # sample with replacement so that it's possible to get zero-sized groups. Get
  # 'num_groups - 1' run ends. The final group will end at 'm'.
  ends_no_final = np.sort(
      np.array(
          [np.random.randint(low=0, high=m) for _ in range(num_groups - 1)],
          dtype=np.int32,
      ),)
  ends = np.concatenate([ends_no_final, np.array([m], dtype=np.int32)])

  # Calculate the run starts by shifting ends 1 to the right. The first run
  # starts at zero.
  starts = np.concatenate([np.zeros(1, dtype=np.int32), ends_no_final])
  return torch.from_numpy(ends - starts).to(torch.int32)


def _trace_kernel_payload(lhs: torch.Tensor, rhs: torch.Tensor,
                          group_sizes: torch.Tensor):
  jax_import_guard()
  import jax
  from jax.experimental.pallas.ops.tpu.megablox import gmm
  from torch_xla.experimental.custom_kernel import trace_pallas
  payload, _ = trace_pallas(gmm, lhs, rhs, group_sizes)

  return payload


def _tolerances(lhs_dtype: torch.dtype, rhs_dtype: torch.dtype,
                out_dtype: torch.dtype) -> tuple[float, float]:
  if (lhs_dtype == torch.bfloat16 or rhs_dtype == torch.bfloat16 or
      out_dtype == torch.bfloat16):
    return 1e-3, 1e-2  # atol, rtol
  return 1e-4, 1e-2  # atol, rtol


LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]


def gmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    group_sizes: torch.Tensor,
    preferred_element_type: torch.dtype = torch.float32,
    tiling: Optional[Union[tuple[int, int, int], LutFn]] = (128, 128, 128),
    group_offset: Optional[torch.Tensor] = None,
    existing_out: Optional[torch.Tensor] = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
):
  payload = _trace_kernel_payload(lhs, rhs, group_sizes)
  out = g.gmm(lhs, rhs, group_sizes, payload, preferred_element_type, tiling,
              group_offset, existing_out, transpose_rhs, interpret)
  return out


def _init_test_cases():
  tests_cases = []
  tests_cases.append({
      'dtype': torch.float32,
      'm': 128,
      'k': 128,
      'n': 128,
      'num_groups': 1
  })
  tests_cases.append({
      'dtype': torch.float32,
      'm': 256,
      'k': 128,
      'n': 128,
      'num_groups': 1
  })
  tests_cases.append({
      'dtype': torch.float32,
      'm': 128,
      'k': 256,
      'n': 128,
      'num_groups': 8
  })
  tests_cases.append({
      'dtype': torch.float32,
      'm': 512,
      'k': 128,
      'n': 256,
      'num_groups': 2
  })
  tests_cases.append({
      'dtype': torch.bfloat16,
      'm': 128,
      'k': 128,
      'n': 128,
      'num_groups': 1
  })
  tests_cases.append({
      'dtype': torch.bfloat16,
      'm': 256,
      'k': 128,
      'n': 128,
      'num_groups': 1
  })
  tests_cases.append({
      'dtype': torch.bfloat16,
      'm': 128,
      'k': 256,
      'n': 128,
      'num_groups': 8
  })
  tests_cases.append({
      'dtype': torch.bfloat16,
      'm': 512,
      'k': 128,
      'n': 256,
      'num_groups': 2
  })
  return tests_cases


if __name__ == '__main__':
  tests_cases = _init_test_cases()
  seed = 421

  for test_case in tests_cases:
    print("Test Case: ", test_case)
    num_groups = test_case['num_groups']
    k = test_case['k']
    m = test_case['m']
    n = test_case['n']
    lhs_dtype = rhs_dtype = test_case['dtype']
    out_dtype = torch.float32

    torch.random.manual_seed(seed)
    lhs = torch.rand(m, k, dtype=lhs_dtype).to('xla')
    rhs = torch.rand(num_groups, k, n, dtype=rhs_dtype).to('xla')
    group_sizes = _group_sizes_strategy(m=m, num_groups=num_groups)
    out = gmm(lhs, rhs, group_sizes)

    ref_out = _reference_gmm(
        lhs.to('cpu').float().numpy(),
        rhs.to('cpu').float().numpy(), group_sizes.numpy())

    atol, rtol = _tolerances(lhs_dtype, rhs_dtype, out_dtype)
    np.testing.assert_allclose(
        ref_out, np.array(out[0].to('cpu')), rtol=rtol, atol=atol)
