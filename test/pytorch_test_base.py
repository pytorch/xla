import collections
import copy
import os
import re
import sys
import runpy

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu

DEFAULT_FLOATING_PRECISION = 1e-3

TORCH_TEST_PRECIIONS = {
    # test_name : floating_precision,
    'test_pow_xla_float32': 0.0035,
    'test_pow_xla_float64': 0.0045,
    'test_var_neg_dim_xla_bfloat16': 0.01,
    'test_sum_xla_bfloat16': 0.1,
}

DISABLED_TORCH_TESTS_ANY = {
    # test_torch.py
    'TestDevicePrecisionXLA': {
        'test_sum_cpu_device_mismatch',  # doesn't raise
        'test_solve_methods_arg_device',  # doesn't raise
        'test_min_max_nan',  # XLA min/max ignores Nans.
        'test_min_max_binary_op_nan',  # XLA min/max ignores Nans.
        'test_copy_broadcast',
    },
    'TestTensorDeviceOpsXLA': {
        'test_block_diag_scipy',  #FIXME: RuntimeError: Error while lowering: f32[1,6]{1,0} xla::unselect, dim=1, start=2, end=2, stride=0
        'test_mean_64bit_indexing_xla',  # protobuf limit exceeded
    },
    'TestTorchDeviceTypeXLA': {
        'test_addmm_sizes',  # FIXME: very slow compile
        'test_addcmul',  # FIXME: complex dtype
        'test_clamp',  # slow
        'test_clamp_propagates_nans_xla',  # XLA min/max ignores Nans.
        'test_cummax_discontiguous',  # Checking contiguity
        'test_cummin_discontiguous',  # Checking contiguity
        'test_discontiguous_out_cumsum',  # Checking contiguity
        'test_lu_unpack',  # very slow compile
        'test_view',  # doesn't raise
        'test_sub_typing',  # doesn't raise
        'test_reduction_empty',  # runtime error
        'test_randperm',
        'test_pow',
        'test_pow_scalar_overloads_mem_overlap',  # doesn't raise
        'test_pdist_norm',
        'test_pdist_norm_backward_xla',  # pdist_single
        'test_pdist_norm_forward_xla',  # pdist_single
        'test_nuclear_norm_axes_small_brute_force',
        'test_mul_intertype_scalar',
        'test_masked_select_discontiguous',  # FIXME: wrong result
        'test_memory_format_type',
        'test_memory_format_type_shortcuts',
        'test_memory_format_to',
        'test_memory_format_factory_like_functions_preserve_strides',
        'test_memory_format_empty_like',
        'test_memory_format_clone',
        'test_memory_format_factory_like_functions_preserve',  # assertion error
        'test_memory_format_propagation_rules',  # assert memory format
        'test_max',  # FIXME: XLA min/max ignores NaNs.
        'test_min',  # FIXME: XLA min/max ignores NaNs.
        'test_min_max_binary_op_nan',
        'test_minmax_illegal_dtype',  # Checking runtime error
        'test_argminmax_multiple',  # FIXME: XLA argmin/argmax ignores NaNs.
        'test_mm_xla_bfloat16',  # FIXME: AssertionError: tensor(0.0625) not less than or equal to 0.001
        'test_lu_solve_batched_non_contiguous',
        'test_linspace_xla',  # Takes forever due to inlined sliced equality tests over 1M elements.
        'test_lstsq',
        'test_is_set_to',
        'test_inverse',
        'test_empty_tensor_props',  # stride
        'test_dist',
        'test_dim_function_empty',
        'test_diagflat',
        'test_cat_out',  # doesn't raise
        'test_cumsum',
        'test_copy_mem_overlap',  # doesn't raise
        'test_copy_all_dtypes_and_devices',
        'test_cholesky_solve_batched_non_contiguous',
        'test_cdist_norm',
        'test_cdist_norm_batch',
        'test_cdist_large',
        'test_cdist_large_batch',
        'test_cdist_non_contiguous',
        'test_cdist_non_contiguous_batch',
        'test_broadcast_batched_matmul',  # incorrect Size
        'test_bincount',
        'test_view_all_dtypes_and_devices',  # uses half
        'test_unfold_all_devices_and_dtypes',  # uses half
        'test_tensor_pow_tensor',  # lowering
        'test_tensor_factories_empty',  # uses half
        'test_symeig',
        'test_svd',
        'test_svd_no_singularvectors',
        'test_svd_lowrank',
        'test_pca_lowrank',
        'test_lobpcg_basic',
        'test_lobpcg_ortho',
        'test_lobpcg_scipy',
        'test_lobpcg_torchscript',
        'test_storage_device',  # storage
        'test_roll',
        'test_resize_as_all_dtypes_and_devices',  # uses half
        'test_resize_all_dtypes_and_devices',  # uses half
        'test_pinverse',  # lowering
        'test_norm',
        'test_multinomial',
        'test_multinomial_alias',
        'test_masked_select',  # uses half
        'test_masked_fill_bool_tensor',  # lowering
        'test_lu',
        'test_logical_and',  # uses half
        'test_logical_or',  # uses half
        'test_logical_xor',  # uses half
        'test_logical',  # uses half
        'test_logical_not',  # uses half
        'test_is_signed',  # uses half
        'test_has_storage_numpy',  # storage
        'test_float_scalar_pow_float_tensor',  # lowering
        'test_flip',  # runtime error
        'test_fill_all_dtypes_and_devices',  # uses half
        'test_eye',  # uses half
        'test_empty_strided',  # storage
        'test_dlpack_conversion',  # storage
        'test_dim_reduction',
        'test_det_logdet_slogdet_batched',  # lowering
        'test_ctor_with_numpy_array',  # uses half
        'test_contiguous',  # storage
        'test_clone_all_dtypes_and_devices',  # uses half
        'test_cat_all_dtypes_and_devices',  # uses half
        'test_broadcast',
        'test_bitwise_not',
        'test_advancedindex',  # storage
        'test_add',  # runtime error
        'test_sign',
        'test_qr',  # slow
        'test_std_mean_some_dims',  # slow
        'test_det_logdet_slogdet',  # very slow compile
        'test_matrix_rank',  # slow
        'test_triu_tril',
        'test_stft',  # librosa (?!?) missing
        'test_strided_mismatched_stride_shape',  # Checking runtime error
        'test_strides_propagation',  # Strides
        'test_tensor_shape_empty',  # LLVM OOM in CI
        'test_cholesky_inverse',  # precision (1e-6)
        'test_cholesky_solve_batched',  # precision (2e-12)
        'test_cholesky_solve',  # precision(1e-12)
        'test_lu_solve_batched',  # precision(1e-12)
        'test_lu_solve',  # precision(1e-12)
        'test_solve_batched',  # precision(1e-12)
        'test_solve',  # precision(1e-12)
        'test_triangular_solve_batched',  # precision(3e-12)
        'test_triangular_solve',  # precision (4e-12)
        'test_scalar_check',  # runtime error
        'test_argminmax_large_axis',  # OOM, and the test is grepping "memory" in the exception message
        'test_randn_xla_float32',  # xla doesn't support manual_seed, as_stride
        'test_randn_xla_float64',  # xla doesn't support manual_seed, as_stride
        'test_rand_xla_float32',  # xla doesn't support manual_seed, as_stride
        'test_rand_xla_float64',  # xla doesn't support manual_seed, as_stride
        'test_normal',  # AssertionError: 0.22364577306378963 not less than or equal to 0.2
        'test_uniform_from_to',  # Checks for error strings.
        'test_index_fill_xla',  # half support
        'test_dim_arg_reduction_scalar_xla',  # access dim 0 of scalar tensors
        'test_storage',  # Storage
        'test_deepcopy',  # Storage
        'teeet_deepcopy_scalar',  # Storage
        'test_scatter_different_types',  # Expecting a runtime error
        'test_bernoulli_mem_overlap',  # doesn't raise
        'test_cat_mem_overlap',  # doesn't raise
        'test_gather_mem_overlap',  # doesn't raise
        'test_index_put_mem_overlap',  # doesn't raise
        'test_index_select_mem_overlap',  # doesn't raise
        'test_linlogspace_mem_overlap',  # doesn't raise
        'test_masked_fill_mem_overlap',  # doesn't raise
        'test_masked_scatter_mem_overlap',  # doesn't raise
        'test_masked_select_mem_overlap',  # doesn't raise
        'test_scatter_mem_overlap',  # doesn't raise
        'test_index_mem_overlap',  # doesn't raise
        'test_maximum_minimum_complex',  # doesn't raise
        'test_maximum_minimum_float_xla_bfloat16',  # precision
        'test_maximum_minimum_type_promotion_xla_.*bfloat16.*',  # doesn't raise
        'test_index_add_mem_overlap',  # doesn't raise
        'test_shift_mem_overlap',  # doesn't raise
        'test_muldiv_scalar_xla_bfloat16',  # FIXME
        'test_random_from_to_bool',  # doesn't raise
        'test_random_from_to_xla',  # doesn't raise
        'test_random_to_xla',  # doesn't raise
    },

    # test_view_ops.py
    'TestViewOpsXLA': {
        'test_contiguous_nonview',
        'test_expand_as_view',
        'test_expand_view',
        'test_reshape_nonview',
        'test_unfold_view',
    },

    # test_indexing.py
    'TestIndexingXLA': {
        'test_setitem_expansion_error',  # expecting a different runtime error
        'test_multiple_byte_mask',  # expecting a different runtime error
        'test_empty_slice',  # stride
        'test_byte_tensor_assignment',  # expecting a different runtime error
        'test_byte_mask',  # expecting a different runtime error
        'test_byte_mask_accumulate',  # expecting a different runtime error
        'test_bool_indices',  # expecting a different runtime error
        'test_index_getitem_copy_bools_slices',  # storage
        'test_getitem_scalars',  # storage
        'test_empty_ndim_index',  # expecting a different runtime error
        'test_index_put_byte_indices_xla',  # expecting a different runtime error
    },
    'NumpyTestsXLA': {
        'test_trivial_fancy_out_of_bounds',  # expecting a different runtime error
        'test_boolean_assignment_value_mismatch',  # expecting a different runtime error
        'test_empty_tuple_index',  # storage
        'test_empty_fancy_index',  # expecting a different runtime error
        'test_ellipsis_index',  # storage
        'test_broaderrors_indexing',  # expecting a different runtime error
        'test_boolean_shape_mismatch',  # expecting a different runtime error
        'test_boolean_indexing_weirdness',  # expecting a different runtime error
        'test_boolean_indexing_weirdness_tensors',  # expecting a different runtime error
    },

    # test_nn.py
    'TestNNDeviceTypeXLA': {
        'test_clip_grad_norm_error_if_nonfinite_xla',  # FIXME: edge case of norm
        'test_embedding_backward',  # sparse
        'test_embedding_dense_grad',  # slow
        'test_batchnorm_grad',  # FIXME! UndefinedTensorImpl::_singleton
        'test_pool_invalid_size',  # expecting a different runtime error
        'test_nonlinearity_propagate_nan',  # relu6 with a nan tensor returns a tensor([0.]) instead of a nan tensor
        'test_InstanceNorm3d_general',  # precision (1e-2)
        'test_InstanceNorm2d_general',  # precision (1e-2)
        'test_InstanceNorm1d_general',  # precision (1e-2)
        'test_EmbeddingBag_per_sample_weights_failures',  # expecting a different runtime error
        'test_variable_sequence',  # PackedSequence batch_sizes.device.type should be CPU but is XLA
        'test_embedding_bag_device',  # FIXME! Unsupported device type for sparse layout: xla
        'test_batchnorm_eval',  # FIXME! UndefinedTensorImpl::_singleton
        'test_MaxPool2d_indices',  # lowering
        'test_MaxPool1d_indices',  # lowering
        'test_EmbeddingBag_per_sample_weights_and_no_offsets',  # FIXME! Unsupported device type for sparse layout: xla
        'test_softshrink_negative',  # runtime error
        'test_nll_loss_empty_tensor_reduction_mean',  # floating point division 0 by 0, expecting nan but get 0
        'test_fold',  # The gradient check code errors out on type() call, and code is slow on XLA
        'test_unfold',  # The gradient check code errors out on type() call, and code is slow on XLA
        'test_hardsigmoid_grad_xla',  # gradient check is slow
        'test_leaky_relu_inplace_overlap_xla',  # doesn't raise
        'test_threshold_inplace_overlap_xla',  # doesn't raise
        'test_elu_inplace_overlap_xla',  # doesn't raise
        'test_hardswish_inplace_overlap_xla',  # doesn't raise
        'test_silu_inplace_overlap_xla',  # doesn't raise
        'test_softplus_inplace_overlap_xla',  # doesn't raise
        'test_softshrink_inplace_overlap_xla',  # doesn't raise
    },

    # test_type_promotion.py
    'TestTypePromotionXLA': {
        'test_many_promotions',  # stride
        'test_inplace',  # expecting a different runtime error
        'test_indexing',  # expecting a different runtime error
        'test_alternate_result',  # expecting a different runtime error
        'test_half',  # half support
        'test_complex_promotion',  # complex support
        'test_complex_scalar_mult_tensor_promotion',  # complex support
        'test_div_promotion_inplace_xla',  # doesn't raise
    }
}

DISABLED_TORCH_TESTS_TPU_ONLY = {
    # test_torch.py
    'TestDevicePrecisionXLA': {
        'test_digamma',  # Precision issue at the first assert, then NAN handling (both on TPU)
    },
    'TestTensorDeviceOpsXLA': {
        'test_pow_inplace_xla',  # (TPU) 0.0032 vs 0.001
        'test_pow_inplace_3_xla',  # (TPU) 0.0028 vs 0.001
        'test_pow_3_xla',  # (TPU) 0.0028 vs 0.001
        'test_pow_-2_xla',  # (TPU) 0.391 vs 0.001
        'test_topk_dim_sort_xla',  # (TPU) unimplemented HLO for X64
        'test_topk_dim_desc_sort_xla',  # (TPU) unimplemented HLO for X64
        'test_sort_xla',  # (TPU) unimplemented HLO for X64
        'test_sort_neg_dim_xla',  # (TPU) unimplemented HLO for X64
        'test_sort_neg_dim_descending_xla',  # (TPU) unimplemented HLO for X64
        'test_sort_dim_xla',  # (TPU) unimplemented HLO for X64
        'test_sort_dim_descending_xla',  # (TPU) unimplemented HLO for X64
        'test_kthvalue_xla',  # (TPU) unimplemented HLO for X64
        'test_kthvalue_neg_dim_xla',  # (TPU) unimplemented HLO for X64
        'test_kthvalue_dim_xla',  # (TPU) unimplemented HLO for X64
        'test_eig_with_eigvec_xla_float64',  # Precision: tensor(1.1798, dtype=torch.float64) not less than or equal to 0.001
        'test_cumprod_xla',  # FIXME: TPU X64Rewriter doesn't support reduce-window
        'test_cumprod_neg_dim_xla',  # FIXME: TPU X64Rewriter doesn't support reduce-window
        'test_topk_neg_dim_sort_xla',  # (TPU) unimplemented HLO for X64
        'test_clamp_min_xla_float64',  # float64 limit, TPU does not have real F64
        'test_clamp_min_inplace_xla_float64',  # float64 limit, TPU does not have real F64
        'test_clamp_max_xla_float64',  # float64 limit, TPU does not have real F64
        'test_clamp_max_inplace_xla_float64',  # float64 limit, TPU does not have real F64
    },
    'TestTorchDeviceTypeXLA': {
        'test_cholesky_solve_batched_broadcasting',  # (TPU) 0.0039 vs 0.001
        'test_cholesky_solve_batched_many_batches',  # (TPU) 0.36 vs 0.001
        'test_triangular_solve_batched_many_batches',  # (TPU) 1.02 vs 0.001
        'test_triangular_solve_batched_broadcasting',  # (TPU) 1.5 vs 0.001
        'test_random_from_to_xla_int32',  # precision, TPU does not have real F64
        'test_uniform_from_to_xla_float64',  # float64 limit, TPU does not have real F64
        'test_topk_integral_xla_int64',  # (TPU) unimplemented HLO for X64
        'test_float_to_int_conversion_finite_xla',  # different behavior than numpy when casting float_max/min to int types
        'test_block_diag_scipy',  # failed to handle np.complex128 as input to tensor.
        'test_remainder_fmod_large_dividend_xla',  # precision, remainder with 1e9 gives incorrect answer
        'test_logical_not_out_xla',  # constant with type f16 and f64 is not supported
        'test_i0_range1_xla_bfloat16',  # precision, 52480.0 vs. 54016.0
        'test_i0_range2_xla_bfloat16',  # precision, 7.019985352739087e+36 vs. 5.815372481559007e+36
        'test_bucketization_xla',  # server side crash
        'test_median_real_values_xla_int64',  # TPU X64Rewriter doesn't support sort
        'test_copysign_xla.*bfloat16.*',  # precision
        'test_nondeterministic_alert_bincount_xla',  # server side crash
        'test_nondeterministic_alert_histc_xla',  # server side crash
        'test_nondeterministic_alert_grid_sample_2d_xla',  # server side crash
        'test_nondeterministic_alert_grid_sample_3d_xla',  # server side crash
        'test_nondeterministic_alert_index_add_xla',  # server side crash
        'test_put_xla_bfloat16',  # (TPU) 0.46484375 vs. 0.484375
        'test_take_xla_bfloat16',  # (TPU) -6.53125 vs. -6.5625
        'test_multinomial_constraints',  # server side crash
        'test_multinomial_invalid_distribution',  # server side crash
        'test_softplus_low_threshold_xla',  # server side crash
        'test_put_xla_float64',  # slow on TPU (~16 min)
        'test_put_xla_int16',  # slow on TPU (~13 min)
        'test_put_xla_int32',  # slow on TPU (~22 min)
        'test_put_xla_int64',  # slow on TPU (~15 min)
        'test_put_xla_int8',  # slow on TPU (~15 min)
    },

    # test_indexing.py
    'TestIndexingXLA': {
        'test_index_put_accumulate_large_tensor_xla',  # memory limit exceeded on v2-8
    },

    # test_nn.py
    'TestNNDeviceTypeXLA': {
        'test_embedding_bag_empty_input_xla',  # server side crash
        'test_EmbeddingBag_empty_per_sample_weights_and_offsets_xla',  # server side crash
        'test_softplus_low_threshold',  # grad check failure
        'test_Dropout',  # too slow
    },

    # test_type_promotion.py
    'TestTypePromotionXLA': {
        'test_bfloat16_xla',  # half support
    }
}

DISABLED_TORCH_TESTS_GPU_ONLY = {
    # test_torch.py
    'TestTorchDeviceTypeXLA': {
        'test_maximum_minimum_float_nan_and_inf',  # maximum(nan,inf) = inf on GPU
    },

    # test_indexing.py
    'TestIndexingXLA': {
        'test_index_put_accumulate_large_tensor_xla',  # illegal memory access was encountered
    },
}


class MatchSet(object):

  def __init__(self):
    self.exact = set()
    self.regex = set()


def prepare_match_set(s):
  ps = dict()
  for k, v in s.items():
    mset = MatchSet()
    for m in v:
      if re.match(r'\w+$', m):
        mset.exact.add(m)
      else:
        mset.regex.add(m)
    ps[k] = mset
  return ps


def match_name(name, mset):
  if name in mset.exact:
    return True
  for m in mset.regex:
    if re.match(m, name):
      return True
  return False


def union_of_disabled_tests(sets):
  union = collections.defaultdict(set)
  for s in sets:
    for k, v in s.items():
      union[k] = union[k] | v
  return union


DISABLED_TORCH_TESTS_CPU = DISABLED_TORCH_TESTS_ANY
DISABLED_TORCH_TESTS_GPU = union_of_disabled_tests(
    [DISABLED_TORCH_TESTS_ANY, DISABLED_TORCH_TESTS_GPU_ONLY])
DISABLED_TORCH_TESTS_TPU = union_of_disabled_tests(
    [DISABLED_TORCH_TESTS_ANY, DISABLED_TORCH_TESTS_TPU_ONLY])

DISABLED_TORCH_TESTS = {
    'TPU': prepare_match_set(DISABLED_TORCH_TESTS_TPU),
    'CPU': prepare_match_set(DISABLED_TORCH_TESTS_CPU),
    'GPU': prepare_match_set(DISABLED_TORCH_TESTS_GPU),
}


class XLATestBase(DeviceTypeTestBase):
  device_type = 'xla'
  unsupported_dtypes = {
      torch.half, torch.complex32, torch.complex64, torch.complex128
  }
  precision = DEFAULT_FLOATING_PRECISION

  @staticmethod
  def _alt_lookup(d, keys, defval):
    for k in keys:
      value = d.get(k, None)
      if value is not None:
        return value
    return defval

  # Overrides to instantiate tests that are known to run quickly
  # and correctly on XLA.
  @classmethod
  def instantiate_test(cls, name, test):
    test_name = name + '_' + cls.device_type
    class_name = cls.__name__
    real_device_type = xm.xla_device_hw(str(xm.xla_device()))
    assert real_device_type in DISABLED_TORCH_TESTS, 'Unsupported device type:' + real_device_type
    disabled_torch_tests = DISABLED_TORCH_TESTS[real_device_type]

    @wraps(test)
    def disallowed_test(self, test=test):
      raise unittest.SkipTest('skipped on XLA')
      return test(self, cls.device_type)

    if class_name in disabled_torch_tests and (
        match_name(test_name, disabled_torch_tests[class_name]) or
        match_name(name, disabled_torch_tests[class_name])):
      assert not hasattr(
          cls, test_name), 'Redefinition of test {0}'.format(test_name)
      setattr(cls, test_name, disallowed_test)
    else:  # Test is allowed
      dtype_combinations = cls._get_dtypes(test)
      if dtype_combinations is None:  # Tests without dtype variants are instantiated as usual
        super().instantiate_test(name, test)
      else:  # Tests with dtype variants have unsupported dtypes skipped
        # Sets default precision for floating types to bfloat16 precision
        if not hasattr(test, 'precision_overrides'):
          test.precision_overrides = {}
        xla_dtypes = []
        for dtype_combination in dtype_combinations:
          if type(dtype_combination) == torch.dtype:
            dtype_combination = (dtype_combination,)
          dtype_test_name = test_name
          skipped = False
          for dtype in dtype_combination:
            dtype_test_name += '_' + str(dtype).split('.')[1]
          for dtype in dtype_combination:
            if dtype in cls.unsupported_dtypes:
              reason = 'XLA does not support dtype {0}'.format(str(dtype))

              @wraps(test)
              def skipped_test(self, *args, reason=reason, **kwargs):
                raise unittest.SkipTest(reason)

              assert not hasattr(
                  cls, dtype_test_name), 'Redefinition of test {0}'.format(
                      dtype_test_name)
              setattr(cls, dtype_test_name, skipped_test)
              skipped = True
              break
            if dtype in [torch.float, torch.double, torch.bfloat16]:
              floating_precision = XLATestBase._alt_lookup(
                  TORCH_TEST_PRECIIONS,
                  [dtype_test_name, test_name, test.__name__],
                  DEFAULT_FLOATING_PRECISION)
              if dtype not in test.precision_overrides or test.precision_overrides[
                  dtype] < floating_precision:
                test.precision_overrides[dtype] = floating_precision

          if class_name in disabled_torch_tests and match_name(
              dtype_test_name, disabled_torch_tests[class_name]):
            skipped = True
            setattr(cls, dtype_test_name, disallowed_test)
          if not skipped:
            xla_dtypes.append(
                dtype_combination
                if len(dtype_combination) > 1 else dtype_combination[0])
        if len(xla_dtypes) != 0:
          test.dtypes[cls.device_type] = xla_dtypes
          super().instantiate_test(name, test)

  @classmethod
  def get_primary_device(cls):
    return cls.primary_device

  @classmethod
  def setUpClass(cls):
    # Sets the primary test device to the xla_device (CPU or TPU)
    cls.primary_device = str(xm.xla_device())
    torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
        use_full_mat_mul_precision=True)

  def setUp(self):
    super().setUp()
    xm.set_rng_state(101)

  def prepare_for_compare(self, tx, ty):
    print_tensors = xu.getenv_as('TEST_PRINT_TENSORS', bool, defval=False)
    x, y = tx, ty
    if type(x) == torch.Tensor:
      x = tx.to(device='cpu')
      if print_tensors:
        print('Tensor X ({}):\n{}'.format(tx.device, x), file=sys.stderr)
    if type(y) == torch.Tensor:
      y = ty.to(device='cpu')
      if print_tensors:
        print('Tensor Y ({}):\n{}'.format(ty.device, y), file=sys.stderr)
    return x, y

  def _override_prec(self, args, name):
    prec = args.get(name, None)
    if prec is None:
      args[name] = self.precision
    else:
      args[name] = max(self.precision, prec)
    return args

  def _rewrite_compare_args(self, kwargs):
    rwargs = copy.copy(kwargs)
    rwargs = self._override_prec(rwargs, 'rtol')
    rwargs = self._override_prec(rwargs, 'atol')
    return rwargs

  # Overrides assertEqual to popular custom precision
  def assertEqual(self, x, y, *args, **kwargs):
    # HACK: Handle the dual nature of the assertEqual() PyTorch API whose first
    # argument can be prec (floating) or msg (string).
    if not args or isinstance(args[0], str):
      kwargs = self._rewrite_compare_args(kwargs)
    elif isinstance(args[0], (float, int)):
      args = [max(args[0], self.precision)] + list(args[1:])

    gmode = os.environ.get('TEST_PRINT_GRAPH', '').lower()
    if gmode == 'text':
      if type(x) == torch.Tensor and xm.is_xla_tensor(x):
        print(
            '\nTest Graph (x):\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_text([x])),
            file=sys.stderr)
      if type(y) == torch.Tensor and xm.is_xla_tensor(y):
        print(
            '\nTest Graph (y):\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_text([y])),
            file=sys.stderr)
    elif gmode == 'hlo':
      if type(x) == torch.Tensor and xm.is_xla_tensor(x):
        print(
            '\nTest Graph (x):\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_hlo([x])),
            file=sys.stderr)
      if type(y) == torch.Tensor and xm.is_xla_tensor(y):
        print(
            '\nTest Graph (y):\n{}'.format(
                torch_xla._XLAC._get_xla_tensors_hlo([y])),
            file=sys.stderr)
    elif gmode:
      raise RuntimeError('Invalid TEST_PRINT_GRAPH value: {}'.format(gmode))
    x, y = self.prepare_for_compare(x, y)
    return DeviceTypeTestBase.assertEqual(self, x, y, *args, **kwargs)


TEST_CLASS = XLATestBase
