import torch

import torch_xla.core.xla_model as xm

from microbench import microbench
from bench import do_bench

torch.set_float32_matmul_precision('high')


def generate_gemm_shape(start, end, stride):
  for n in range(start, end + 1, stride):
    lhs_shape = (n, n)
    rhs_shape = (n, n)
    yield lhs_shape, rhs_shape


def get_matmuls(device, dtype, backend):
  matmuls = []
  for lhs_shape, rhs_shape in generate_gemm_shape(
      start=256, end=8192, stride=128):
    lhs_tensor = torch.randn(lhs_shape, device=device, dtype=dtype)
    rhs_tensor = torch.randn(rhs_shape, device=device, dtype=dtype)
    mm = torch.compile(
        lambda lhs_tensor=lhs_tensor, rhs_tensor=rhs_tensor: torch.matmul(
            lhs_tensor, rhs_tensor)[0, 0].item(),
        backend=backend)
    matmuls.append((lhs_tensor.shape, rhs_tensor.shape, mm))
  return matmuls


def get_test_name(dtype, lhs_shape, rhs_shape):
  shape_str = lambda sh: 'x'.join([str(dim) for dim in list(sh)])
  return f"matmul-{dtype}-lhs{shape_str(lhs_shape)}-rhs{shape_str(rhs_shape)}"


def main():
  """Benchmarks squared matrices against each other for both Inductor, and XLA.
  """

  xla_bench_fn = lambda fn: do_bench(
      fn,
      return_mode='min',
      sync_fn=lambda: xm.wait_device_ops(),
      device=xm.xla_device())
  ind_bench_fn = lambda fn: do_bench(
      fn,
      return_mode='min',
      sync_fn=lambda: torch.cuda.synchronize(),
      device='cuda')

  dtypes = [torch.float32, torch.float16, torch.bfloat16]
  for dtype in dtypes:
    for inductor_matmul, xla_matmul in zip(
        get_matmuls(device='cuda', dtype=dtype, backend='inductor'),
        get_matmuls(device=xm.xla_device(), dtype=dtype, backend='openxla')):
      ind_lhs_shape, ind_rhs_shape, ind_fn = inductor_matmul
      xla_lhs_shape, xla_rhs_shape, xla_fn = xla_matmul
      assert ind_lhs_shape == xla_lhs_shape, f"Expect matmul shapes to match for benchmarking. Mismatch lhs: {ind_lhs_shape}, rhs: {xla_rhs_shape}"
      assert ind_rhs_shape == xla_rhs_shape, f"Expect matmul shapes to match for benchmarking. Mistmatch rhs: {ind_rhs_shape}, rhs: {ind_rhs_shape}"

      test_name = get_test_name(dtype, ind_lhs_shape, ind_rhs_shape)
      result = microbench(
          test_name,
          baseline_fn=ind_fn,
          testing_fn=xla_fn,
          baseline_bench_fn=ind_bench_fn,
          testing_bench_fn=xla_bench_fn,
          baseline_sync_fn=torch.cuda.synchronize,
          testing_sync_fn=xm.wait_device_ops)
      print(result)


if __name__ == '__main__':
  main()
