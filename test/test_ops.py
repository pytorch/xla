import collections
import numbers
from typing import Callable

import torch
from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.testing._internal.common_methods_invocations import \
    (SampleInput, op_db)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops)


class AllowedOpInfoEntry(
    collections.namedtuple("AllowedOpInfoEntry",
                           ["name", "variant_test_name"])):

  def __new__(cls, name, variant_test_name=""):
    return super(AllowedOpInfoEntry, cls).__new__(cls, name, variant_test_name)


# Ops (subset of `op_db`) which are known to pass the reference tests on XLA.
allowed_opinfo = set(
    map(
        lambda op: op.name + "_" + op.variant_test_name
        if op.variant_test_name != '' else op.name,
        {
            AllowedOpInfoEntry('abs'),
            AllowedOpInfoEntry('add'),
            AllowedOpInfoEntry('mul'),
            AllowedOpInfoEntry('sub'),
            AllowedOpInfoEntry('addmm'),
            AllowedOpInfoEntry('addmm', 'decomposed'),
            AllowedOpInfoEntry('addmv'),
            AllowedOpInfoEntry('addbmm'),
            AllowedOpInfoEntry('baddbmm'),
            AllowedOpInfoEntry('dot'),
            AllowedOpInfoEntry('vdot'),
            AllowedOpInfoEntry('bmm'),
            AllowedOpInfoEntry('mv'),
            AllowedOpInfoEntry('addr'),
            AllowedOpInfoEntry('addcmul'),
            AllowedOpInfoEntry('addcdiv'),
            AllowedOpInfoEntry('atan2'),
            AllowedOpInfoEntry('broadcast_to'),
            AllowedOpInfoEntry('bitwise_not'),
            AllowedOpInfoEntry('bitwise_left_shift'),
            AllowedOpInfoEntry('bitwise_right_shift'),
            AllowedOpInfoEntry('ceil'),
            AllowedOpInfoEntry('cholesky'),
            AllowedOpInfoEntry('cholesky_inverse'),
            AllowedOpInfoEntry('chunk'),
            AllowedOpInfoEntry('clone'),
            AllowedOpInfoEntry('contiguous'),
            AllowedOpInfoEntry('clamp'),
            AllowedOpInfoEntry('positive'),
            AllowedOpInfoEntry('conj'),
            AllowedOpInfoEntry('conj_physical'),
            AllowedOpInfoEntry('resolve_conj'),
            AllowedOpInfoEntry('view_as_real'),
            AllowedOpInfoEntry('view_as_complex'),
            AllowedOpInfoEntry('complex'),
            AllowedOpInfoEntry('copysign'),
            AllowedOpInfoEntry('cross'),
            AllowedOpInfoEntry('cummax'),
            AllowedOpInfoEntry('cummin'),
            AllowedOpInfoEntry('deg2rad'),
            AllowedOpInfoEntry('div', 'no_rounding_mode'),
            AllowedOpInfoEntry('div', 'trunc_rounding'),
            AllowedOpInfoEntry('div', 'floor_rounding'),
            AllowedOpInfoEntry('true_divide'),
            AllowedOpInfoEntry('expand'),
            AllowedOpInfoEntry('expand_as'),
            AllowedOpInfoEntry('diag'),
            AllowedOpInfoEntry('eq'),
            AllowedOpInfoEntry('fmax'),
            AllowedOpInfoEntry('fmin'),
            AllowedOpInfoEntry('fmod'),
            AllowedOpInfoEntry('fmod', 'autodiffed'),
            AllowedOpInfoEntry('remainder'),
            AllowedOpInfoEntry('remainder', 'autodiffed'),
            AllowedOpInfoEntry('frac'),
            AllowedOpInfoEntry('fft.fft'),
            AllowedOpInfoEntry('fft.fftn'),
            AllowedOpInfoEntry('fft.hfft'),
            AllowedOpInfoEntry('fft.rfft'),
            AllowedOpInfoEntry('fft.rfftn'),
            AllowedOpInfoEntry('fft.ifftn'),
            AllowedOpInfoEntry('fft.irfft'),
            AllowedOpInfoEntry('fft.irfftn'),
            AllowedOpInfoEntry('floor'),
            AllowedOpInfoEntry('flip'),
            AllowedOpInfoEntry('fliplr'),
            AllowedOpInfoEntry('flipud'),
            AllowedOpInfoEntry('i0'),
            AllowedOpInfoEntry('special.i0e'),
            AllowedOpInfoEntry('special.i1'),
            AllowedOpInfoEntry('special.i1e'),
            AllowedOpInfoEntry('special.ndtr'),
            AllowedOpInfoEntry('floor_divide'),
            AllowedOpInfoEntry('frexp'),
            AllowedOpInfoEntry('ge'),
            AllowedOpInfoEntry('geqrf'),
            AllowedOpInfoEntry('gt'),
            AllowedOpInfoEntry('imag'),
            AllowedOpInfoEntry('inverse'),
            AllowedOpInfoEntry('isin'),
            AllowedOpInfoEntry('le'),
            AllowedOpInfoEntry('linalg.det'),
            AllowedOpInfoEntry('linalg.cholesky'),
            AllowedOpInfoEntry('linalg.cholesky_ex'),
            AllowedOpInfoEntry('linalg.eig'),
            AllowedOpInfoEntry('linalg.householder_product'),
            AllowedOpInfoEntry('linalg.matrix_power'),
            AllowedOpInfoEntry('linalg.qr'),
            AllowedOpInfoEntry('linalg.slogdet'),
            AllowedOpInfoEntry('log'),
            AllowedOpInfoEntry('log10'),
            AllowedOpInfoEntry('log1p'),
            AllowedOpInfoEntry('log2'),
            AllowedOpInfoEntry('logaddexp'),
            AllowedOpInfoEntry('logaddexp2'),
            AllowedOpInfoEntry('logical_not'),
            AllowedOpInfoEntry('lt'),
            AllowedOpInfoEntry('lu'),
            AllowedOpInfoEntry('lu_unpack'),
            AllowedOpInfoEntry('masked_fill'),
            AllowedOpInfoEntry('masked_scatter'),
            AllowedOpInfoEntry('masked_select'),
            AllowedOpInfoEntry('matrix_exp'),
            AllowedOpInfoEntry('max', 'binary'),
            AllowedOpInfoEntry('max', 'reduction_no_dim'),
            AllowedOpInfoEntry('median'),
            AllowedOpInfoEntry('nanmedian'),
            AllowedOpInfoEntry('min', 'binary '),
            AllowedOpInfoEntry('min', 'reduction_no_dim'),
            AllowedOpInfoEntry('nansum'),
            AllowedOpInfoEntry('quantile'),
            AllowedOpInfoEntry('maximum'),
            AllowedOpInfoEntry('minimum'),
            AllowedOpInfoEntry('nn.functional.hardswish'),
            AllowedOpInfoEntry('nn.functional.leaky_relu'),
            AllowedOpInfoEntry('nn.functional.hardshrink'),
            AllowedOpInfoEntry('nn.functional.hardtanh'),
            AllowedOpInfoEntry('nn.functional.gelu'),
            AllowedOpInfoEntry('nn.functional.relu6'),
            AllowedOpInfoEntry('mm'),
            AllowedOpInfoEntry('mode'),
            AllowedOpInfoEntry('polygamma', 'polygamma_n_0'),
            AllowedOpInfoEntry('mvlgamma', 'mvlgamma_p_1'),
            AllowedOpInfoEntry('ne'),
            AllowedOpInfoEntry('narrow'),
            AllowedOpInfoEntry('neg'),
            AllowedOpInfoEntry('dist'),
            AllowedOpInfoEntry('outer'),
            AllowedOpInfoEntry('ormqr'),
            AllowedOpInfoEntry('permute'),
            AllowedOpInfoEntry('pow'),
            AllowedOpInfoEntry('float_power'),
            AllowedOpInfoEntry('qr'),
            AllowedOpInfoEntry('rad2deg'),
            AllowedOpInfoEntry('real'),
            AllowedOpInfoEntry('roll'),
            AllowedOpInfoEntry('rot90'),
            AllowedOpInfoEntry('round'),
            AllowedOpInfoEntry('sinc'),
            AllowedOpInfoEntry('sign'),
            AllowedOpInfoEntry('sgn'),
            AllowedOpInfoEntry('split'),
            AllowedOpInfoEntry('split', 'list_args'),
            AllowedOpInfoEntry('split_with_sizes'),
            AllowedOpInfoEntry('__radd__'),
            AllowedOpInfoEntry('__rmul__'),
            AllowedOpInfoEntry('__rpow__'),
            AllowedOpInfoEntry('__rsub__'),
            AllowedOpInfoEntry('rsub', 'rsub_tensor'),
            AllowedOpInfoEntry('select'),
            AllowedOpInfoEntry('signbit'),
            AllowedOpInfoEntry('solve'),
            AllowedOpInfoEntry('tensor_split'),
            AllowedOpInfoEntry('hsplit'),
            AllowedOpInfoEntry('vsplit'),
            AllowedOpInfoEntry('dsplit'),
            AllowedOpInfoEntry('triangular_solve'),
            AllowedOpInfoEntry('trunc'),
            AllowedOpInfoEntry('exp2'),
            AllowedOpInfoEntry('nan_to_num'),
            AllowedOpInfoEntry('square'),
            AllowedOpInfoEntry('lerp'),
            AllowedOpInfoEntry('linalg.inv'),
            AllowedOpInfoEntry('linalg.inv_ex'),
            AllowedOpInfoEntry('angle'),
            AllowedOpInfoEntry('linalg.solve'),
            AllowedOpInfoEntry('linalg.matrix_rank'),
            AllowedOpInfoEntry('linalg.svd'),
            AllowedOpInfoEntry('linalg.svdvals'),
            AllowedOpInfoEntry('polar'),
            AllowedOpInfoEntry('ravel'),
            AllowedOpInfoEntry('reshape'),
            AllowedOpInfoEntry('reshape_as'),
            AllowedOpInfoEntry('view'),
            AllowedOpInfoEntry('view_as'),
            AllowedOpInfoEntry('put'),
            AllowedOpInfoEntry('take'),
            AllowedOpInfoEntry('stack'),
            AllowedOpInfoEntry('hstack'),
            AllowedOpInfoEntry('hypot'),
            AllowedOpInfoEntry('histogram'),
            AllowedOpInfoEntry('vstack'),
            AllowedOpInfoEntry('dstack'),
            AllowedOpInfoEntry('unfold'),
            AllowedOpInfoEntry('msort'),
            AllowedOpInfoEntry('movedim'),
            AllowedOpInfoEntry('renorm'),
            AllowedOpInfoEntry('fill_'),
            AllowedOpInfoEntry('resize_'),
            AllowedOpInfoEntry('resize_as_'),
            AllowedOpInfoEntry('take_along_dim'),
            AllowedOpInfoEntry('unsqueeze'),
            AllowedOpInfoEntry('xlogy'),
            AllowedOpInfoEntry('zero_'),
            AllowedOpInfoEntry('special.xlog1py'),
            AllowedOpInfoEntry('special.zeta'),
            AllowedOpInfoEntry('special.zeta', 'grad'),
            AllowedOpInfoEntry('trace'),
            AllowedOpInfoEntry('tril'),
            AllowedOpInfoEntry('triu'),
            AllowedOpInfoEntry('kron'),
            AllowedOpInfoEntry('inner'),
            AllowedOpInfoEntry('tensordot'),
            AllowedOpInfoEntry('logcumsumexp'),
            AllowedOpInfoEntry('digamma'),
            AllowedOpInfoEntry('special.entr'),
            AllowedOpInfoEntry('special.ndtri'),
            AllowedOpInfoEntry('lgamma'),
            AllowedOpInfoEntry('log_softmax'),
            AllowedOpInfoEntry('logit'),
            AllowedOpInfoEntry('where'),
            AllowedOpInfoEntry('norm', 'fro'),
            AllowedOpInfoEntry('special.erfcx'),
            AllowedOpInfoEntry('_native_batch_norm_legit'),

            # Duplicate Redundant entries for this test.
            # AllowedOpInfoEntry('polygamma', 'polygamma_n_1'),
            # AllowedOpInfoEntry('polygamma', 'polygamma_n_2'),
            # AllowedOpInfoEntry('polygamma', 'polygamma_n_3'),
            # AllowedOpInfoEntry('polygamma', 'polygamma_n_4'),
            # AllowedOpInfoEntry('mvlgamma', 'mvlgamma_p_3'),
            # AllowedOpInfoEntry('mvlgamma', 'mvlgamma_p_5'),

            # Failing Ops
            # Refer for more info : https://github.com/pytorch/xla/pull/3019#issuecomment-877132385
            # AllowedOpInfoEntry('einsum'), https://github.com/pytorch/xla/issues/4052
            # AllowedOpInfoEntry('cdist'),  // precision issue on TPU
            # AllowedOpInfoEntry('linalg.multi_dot'),  // failing on CPU
            # AllowedOpInfoEntry('matmul'),            // failing on CPU
            # AllowedOpInfoEntry('__rmatmul__'),       // failing on CPU
            # AllowedOpInfoEntry('linalg.eigvals'),  // failing on TPU
            # AllowedOpInfoEntry('nanquantile'), // TODO: retried at head once xlogy pr merged
            # AllowedOpInfoEntry('amax'),
            # AllowedOpInfoEntry('amin'),
            # AllowedOpInfoEntry('norm', 'nuc'),
            # AllowedOpInfoEntry('norm', 'nuc'),
            # AllowedOpInfoEntry('norm', 'inf'),
            # AllowedOpInfoEntry('max', 'reduction_with_dim'),
            # AllowedOpInfoEntry('min', 'reduction_with_dim'),
            # AllowedOpInfoEntry('log_softmax', 'dtype'),
            # AllowedOpInfoEntry('linalg.matrix_rank', 'hermitian'),
            # AllowedOpInfoEntry('linalg.pinv', 'hermitian'),
            # AllowedOpInfoEntry('clamp', 'scalar'),
            # AllowedOpInfoEntry('acos'),
            # AllowedOpInfoEntry('acosh'),
            # AllowedOpInfoEntry('argmax')
            # AllowedOpInfoEntry('argmin')
            # AllowedOpInfoEntry('asin'),
            # AllowedOpInfoEntry('asinh'),
            # AllowedOpInfoEntry('atan'),
            # AllowedOpInfoEntry('atanh'),
            # AllowedOpInfoEntry('symeig'),
            # AllowedOpInfoEntry('cos'),
            # AllowedOpInfoEntry('cosh'),
            # AllowedOpInfoEntry('cov'),
            # AllowedOpInfoEntry('cumsum'),
            # AllowedOpInfoEntry('cumprod'),
            # AllowedOpInfoEntry('diff'),
            # AllowedOpInfoEntry('exp'),
            # AllowedOpInfoEntry('diag_embed'),
            # AllowedOpInfoEntry('diagonal'),
            # AllowedOpInfoEntry('fft.ifft'),
            # AllowedOpInfoEntry('fft.ihfft'),
            # AllowedOpInfoEntry('gradient'),
            # AllowedOpInfoEntry('kthvalue'),
            # AllowedOpInfoEntry('linalg.cond'),
            # AllowedOpInfoEntry('linalg.eigh'),
            # AllowedOpInfoEntry('linalg.eigvalsh'),
            # AllowedOpInfoEntry('linalg.lstsq'),
            # AllowedOpInfoEntry('linalg.norm'),
            # AllowedOpInfoEntry('linalg.matrix_norm'),
            # AllowedOpInfoEntry('linalg.vector_norm'),
            # AllowedOpInfoEntry('std_mean'),
            # AllowedOpInfoEntry('sum'),
            # AllowedOpInfoEntry('mean'),
            # AllowedOpInfoEntry('topk'),
            # AllowedOpInfoEntry('prod'),
            # AllowedOpInfoEntry('sin'),
            # AllowedOpInfoEntry('sinh'),
            # AllowedOpInfoEntry('__rdiv__'),
            # AllowedOpInfoEntry('__rmod__'),
            # AllowedOpInfoEntry('std'),
            # AllowedOpInfoEntry('tan'),
            # AllowedOpInfoEntry('tanh'),
            # AllowedOpInfoEntry('expm1'),
            # AllowedOpInfoEntry('reciprocal'),
            # AllowedOpInfoEntry('rsqrt'),
            # AllowedOpInfoEntry('sqrt'),
            # AllowedOpInfoEntry('linalg.pinv'),
            # AllowedOpInfoEntry('eig'),
            # AllowedOpInfoEntry('svd'),
            # AllowedOpInfoEntry('pinverse'),
            # AllowedOpInfoEntry('gather'),
            # AllowedOpInfoEntry('index_fill'),
            # AllowedOpInfoEntry('index_copy'),
            # AllowedOpInfoEntry('index_select'),
            # AllowedOpInfoEntry('index_add'),
            # AllowedOpInfoEntry('__getitem__'),
            # AllowedOpInfoEntry('sort'),
            # AllowedOpInfoEntry('scatter'),
            # AllowedOpInfoEntry('scatter_add'),
            # AllowedOpInfoEntry('repeat'),
            # AllowedOpInfoEntry('squeeze'),
            # AllowedOpInfoEntry('tile'),
            # AllowedOpInfoEntry('var'),
            # AllowedOpInfoEntry('logsumexp'),
            # AllowedOpInfoEntry('transpose'),
            # AllowedOpInfoEntry('to_sparse'),
            # AllowedOpInfoEntry('sigmoid'),
            # AllowedOpInfoEntry('erf'),
            # AllowedOpInfoEntry('erfc'),
            # AllowedOpInfoEntry('erfinv'),
            # AllowedOpInfoEntry('norm'),
            # AllowedOpInfoEntry('t'),
            # AllowedOpInfoEntry('logdet'), xla::lodget does not handle empty input

            # Failed on CUDA CI only (investigate)
            # app.circleci.com/pipelines/github/pytorch/xla/9088/workflows/2d59c649-db2b-4384-921e-5e43eba1b51a/jobs/17875
            # AllowedOpInfoEntry('index_put'),

            # Worked locally (but failing on CI both CPU and CUDA)
            # app.circleci.com/pipelines/github/pytorch/xla/9130/workflows/71c74f3d-1735-4328-81b5-784d6e6744da/jobs/17998
            # AllowedOpInfoEntry('var_mean'),
        }))


def is_in_allowed(op):
  name = op.name
  if op.variant_test_name != '':
    name = op.name + "_" + op.variant_test_name

  return name in allowed_opinfo


ops_to_test = list(filter(lambda op: is_in_allowed(op), op_db))


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

        # Passthrough None because some functions wrapped with type promotion
        # wrapper might have optional args
        if x is None:
          return None
        raise ValueError("Unknown type {0}!".format(type(x)))

      cpu_sample_input, cpu_args, cpu_kwargs = to_cpu(sample.input), to_cpu(
          sample.args), to_cpu(sample.kwargs)
      return (cpu_sample_input, cpu_args, cpu_kwargs)

    t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
    cpu_inp, cpu_args, cpu_kwargs = cpu(sample_input)

    actual = torch_fn(t_inp, *t_args, **t_kwargs)
    expected = torch_fn(cpu_inp, *cpu_args, **cpu_kwargs)

    self.assertEqual(actual, expected, exact_dtype=True, exact_device=False)

  @ops(ops_to_test, allowed_dtypes=(torch.float32, torch.long))
  def test_reference_eager(self, device, dtype, op):
    if self.device_type != 'xla':
      self.skipTest("This test runs only on XLA")

    sample_inputs = op.sample_inputs(device, dtype)
    for sample_input in sample_inputs:
      self.compare_with_eager_reference(op, sample_input)


instantiate_device_type_tests(TestOpInfo, globals())

if __name__ == '__main__':
  run_tests()
