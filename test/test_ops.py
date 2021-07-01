import collections
import numbers
from typing import Callable

import torch
from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.testing._internal.common_methods_invocations import \
    (SampleInput, op_db, SkipInfo)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops)

# Skips which are frequently used.
reference_eager_all_skip = SkipInfo(
    "TestOpInfo", "test_reference_eager", device_type='xla')

reference_eager_int_skip = SkipInfo(
    "TestOpInfo",
    "test_reference_eager",
    device_type='xla',
    dtypes=(torch.int64,))

reference_eager_float_skip = SkipInfo(
    "TestOpInfo",
    "test_reference_eager",
    device_type='xla',
    dtypes=(torch.float,))


class SkipEntry(
    collections.namedtuple("SkipEntry", ["op_name", "variant_name", "skip"])):

  def __new__(cls, op_name="", variant_name="", skip=()):
    return super(SkipEntry, cls).__new__(cls, op_name, variant_name, skip)


def update_skips():
  # Function to add skips for this particular test suite.
  # We do it here so that we don't have to update the
  # Pytorch core which doesn't know about this suite.
  # Note: This function mutates the existing OpInfo entry.
  skips = {
      SkipEntry('__getitem__', skip=reference_eager_all_skip),
      SkipEntry('__rdiv__', skip=reference_eager_int_skip),
      SkipEntry('__rmod__', skip=reference_eager_int_skip),
      SkipEntry('acos', skip=reference_eager_int_skip),
      SkipEntry('acosh', skip=reference_eager_int_skip),
      SkipEntry('argmax', skip=reference_eager_all_skip),
      SkipEntry('argmin', skip=reference_eager_all_skip),
      SkipEntry('asin', skip=reference_eager_int_skip),
      SkipEntry('asinh', skip=reference_eager_int_skip),
      SkipEntry('atan', skip=reference_eager_int_skip),
      SkipEntry('atanh', skip=reference_eager_int_skip),
      SkipEntry('clamp', variant_name='scalar', skip=reference_eager_all_skip),
      SkipEntry('cos', skip=reference_eager_int_skip),
      SkipEntry('cosh', skip=reference_eager_int_skip),
      SkipEntry('cov', skip=reference_eager_all_skip),
      SkipEntry('cumprod', skip=reference_eager_all_skip),
      SkipEntry('cumsum', skip=reference_eager_all_skip),
      SkipEntry('diag_embed', skip=reference_eager_all_skip),
      SkipEntry('diagonal', skip=reference_eager_all_skip),
      SkipEntry('diff', skip=reference_eager_all_skip),
      SkipEntry('eig', skip=reference_eager_float_skip),
      SkipEntry('erf', skip=reference_eager_int_skip),
      SkipEntry('erfc', skip=reference_eager_int_skip),
      SkipEntry('erfinv', skip=reference_eager_int_skip),
      SkipEntry('exp', skip=reference_eager_int_skip),
      SkipEntry('expm1', skip=reference_eager_int_skip),
      SkipEntry('fft.ifft', skip=reference_eager_float_skip),
      SkipEntry('fft.ihfft', skip=reference_eager_float_skip),
      SkipEntry('gather', skip=reference_eager_all_skip),
      SkipEntry('gradient', skip=reference_eager_all_skip),
      SkipEntry('index_add', skip=reference_eager_all_skip),
      SkipEntry('index_copy', skip=reference_eager_all_skip),
      SkipEntry('index_fill', skip=reference_eager_all_skip),
      SkipEntry('index_select', skip=reference_eager_all_skip),
      SkipEntry('kthvalue', skip=reference_eager_all_skip),
      SkipEntry('linalg.cond', skip=reference_eager_float_skip),
      SkipEntry('linalg.eigh', skip=reference_eager_float_skip),
      SkipEntry('linalg.eigvalsh', skip=reference_eager_float_skip),
      # SegFaults!
      SkipEntry('linalg.lstsq', skip=reference_eager_float_skip),
      SkipEntry('linalg.matrix_norm', skip=reference_eager_float_skip),
      SkipEntry(
          'linalg.matrix_rank',
          variant_name='hermitian',
          skip=reference_eager_float_skip),
      SkipEntry('linalg.norm', skip=reference_eager_float_skip),
      SkipEntry('linalg.pinv', skip=reference_eager_float_skip),
      SkipEntry(
          'linalg.pinv',
          variant_name='hermitian',
          skip=reference_eager_float_skip),
      SkipEntry('linalg.vector_norm', skip=reference_eager_float_skip),
      SkipEntry('log10', skip=reference_eager_int_skip),
      SkipEntry('log1p', skip=reference_eager_int_skip),
      SkipEntry('log2', skip=reference_eager_int_skip),
      SkipEntry(
          'log_softmax', variant_name='dtype', skip=reference_eager_all_skip),
      SkipEntry('log', skip=reference_eager_int_skip),
      SkipEntry('logsumexp', skip=reference_eager_float_skip),
      SkipEntry(
          'max',
          variant_name='reduction_with_dim',
          skip=reference_eager_all_skip),
      SkipEntry('mean', skip=reference_eager_float_skip),
      SkipEntry(
          'min',
          variant_name='reduction_with_dim',
          skip=reference_eager_all_skip),
      SkipEntry('norm', variant_name='inf', skip=reference_eager_all_skip),
      SkipEntry('norm', variant_name='nuc', skip=reference_eager_all_skip),
      SkipEntry('norm', skip=reference_eager_all_skip),
      SkipEntry('pinverse', skip=reference_eager_float_skip),
      SkipEntry('prod', skip=reference_eager_all_skip),
      SkipEntry('reciprocal', skip=reference_eager_int_skip),
      SkipEntry('repeat', skip=reference_eager_all_skip),
      SkipEntry('rsqrt', skip=reference_eager_int_skip),
      SkipEntry(
          'rsub', variant_name='rsub_scalar', skip=reference_eager_all_skip),
      SkipEntry('scatter_add', skip=reference_eager_all_skip),
      SkipEntry('scatter', skip=reference_eager_all_skip),
      SkipEntry('sigmoid', skip=reference_eager_int_skip),
      SkipEntry('sin', skip=reference_eager_int_skip),
      SkipEntry('sinh', skip=reference_eager_int_skip),
      SkipEntry('sort', skip=reference_eager_all_skip),
      SkipEntry('sqrt', skip=reference_eager_int_skip),
      SkipEntry('squeeze', skip=reference_eager_all_skip),
      SkipEntry('std_mean', skip=reference_eager_float_skip),
      SkipEntry('std', skip=reference_eager_float_skip),
      SkipEntry('sum', skip=reference_eager_all_skip),
      SkipEntry('svd', skip=reference_eager_float_skip),
      SkipEntry('symeig', skip=reference_eager_float_skip),
      SkipEntry('t', skip=reference_eager_all_skip),
      SkipEntry('tan', skip=reference_eager_int_skip),
      SkipEntry('tanh', skip=reference_eager_int_skip),
      SkipEntry('tile', skip=reference_eager_all_skip),
      SkipEntry('to_sparse', skip=reference_eager_all_skip),
      SkipEntry('topk', skip=reference_eager_all_skip),
      SkipEntry('transpose', skip=reference_eager_all_skip),
      SkipEntry('var', skip=reference_eager_float_skip),
  }

  for op in op_db:
    for op_name, variant_name, skip in skips:
      if op.name == op_name and op.variant_test_name == variant_name:
        if op.skips == ():
          op.skips = (skip,)
        else:
          op.skips += (skip,)


# Add skips to OpInfo (if any) for this test suite.
update_skips()


class TestOpInfo(TestCase):

  def compare_with_eager_reference(self, torch_fn: Callable,
                                   sample_input: SampleInput, **kwargs) -> None:

    def cpu(sample: SampleInput):
      # Similar to `numpy` method on SampleInput.
      # Converts tensors to cpu tensors by calling .detach().cpu() on them
      # Numbers, strings, and bool are preserved as is
      # Lists, tuples and dicts are handled by calling this function recursively
      def to_cpu(x):

        def _cpu(t):
          return t.detach().cpu()

        if isinstance(x, torch.Tensor):
          return _cpu(x)
        elif isinstance(x, list):
          return list(map(to_cpu, x))
        elif isinstance(x, tuple):
          return tuple(map(to_cpu, x))
        elif isinstance(x, dict):
          return {k: to_cpu(v) for k, v in x.items()}
        elif isinstance(x, (numbers.Number, bool, str)):
          return x

        raise ValueError("Unknown type {0}!".format(type(x)))

      cpu_sample_input, cpu_args, cpu_kwargs = to_cpu(sample.input), to_cpu(
          sample.args), to_cpu(sample.kwargs)
      return (cpu_sample_input, cpu_args, cpu_kwargs)

    t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
    cpu_inp, cpu_args, cpu_kwargs = cpu(sample_input)

    actual = torch_fn(t_inp, *t_args, **t_kwargs)
    expected = torch_fn(cpu_inp, *cpu_args, **cpu_kwargs)

    self.assertEqual(actual, expected, exact_dtype=True, exact_device=False)

  @ops(op_db, allowed_dtypes=(torch.float32, torch.long))
  def test_reference_eager(self, device, dtype, op):
    if self.device_type != 'xla':
      self.skipTest("This test runs only on XLA")

    sample_inputs = op.sample_inputs(device, dtype)
    for sample_input in sample_inputs:
      self.compare_with_eager_reference(op, sample_input)


instantiate_device_type_tests(TestOpInfo, globals())

if __name__ == '__main__':
  run_tests()
