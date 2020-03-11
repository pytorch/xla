DEFAULT_FLOATING_PRECISION = 1e-3

torch_test_precisions = {
    # test_name : floating_precision,
}

disabled_torch_tests = {
    # test_torch.py
    # TestDevicePrecision
    'test_sum_cpu_device_mismatch',  # doesn't raise
    'test_solve_methods_arg_device',  # doesn't raise
    'test_min_max_nan',  # XLA min/max ignores Nans.
    'test_min_max_binary_op_nan',  # XLA min/max ignores Nans.
    'test_copy_noncontig',
    'test_copy_broadcast',
    'test_digamma',  # Precision issue at the first assert, then NAN handling (both on TPU)

    # TestTensorDeviceOps
    'test_prod_neg_dim_xla',
    'test_prod_dim_xla',
    'test_cumprod_xla',
    'test_cumprod_neg_dim_xla',
    'test_mean_64bit_indexing_xla',  # protobuf limit exceeded
    'test_pow_xla',  # (TPU) 0.0043 vs 0.001
    'test_pow_xla',  # (TPU) 0.0032 vs 0.001
    'test_pow_inplace_xla',  # (TPU) 0.0043 vs 0.001
    'test_pow_inplace_xla',  # (TPU) 0.0032 vs 0.001
    'test_pow_inplace_3_xla',  # (TPU) 0.0036 vs 0.001
    'test_pow_inplace_3_xla',  # (TPU) 0.0028 vs 0.001
    'test_pow_3_xla',  # (TPU) 0.0036 vs 0.001
    'test_pow_3_xla',  # (TPU) 0.0028 vs 0.001
    'test_pow_-2_xla',  # (TPU) 0.0913 vs 0.001
    'test_pow_-2_xla',  # (TPU) 0.391 vs 0.001
    'test_topk_neg_dim_sort_xla',  # (TPU) unimplemented HLO for X64
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

    # TestTorchDeviceType
    'test_addmm_sizes',  # FIXME: very slow compile
    'test_clamp',  # slow
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
    'test_memory_format_type',
    'test_memory_format_type_shortcuts',
    'test_memory_format_to',
    'test_memory_format_factory_like_functions_preserve_strides',
    'test_memory_format_empty_like',
    'test_memory_format_clone',
    'test_memory_format_factory_like_functions_preserve', # assertion error
    'test_lu_solve_batched_non_contiguous',
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
    'test_tensor_shape_empty',  # LLVM OOM in CI
    'test_cholesky_inverse',  # precision (1e-6)
    'test_cholesky_solve_batched_broadcasting',  # (TPU) 0.0039 vs 0.001
    'test_cholesky_solve_batched_many_batches',  # (TPU) 0.36 vs 0.001
    'test_cholesky_solve_batched',  # (TPU) precision (1e-5)
    'test_cholesky_solve',  # (TPU) precision (1e-5)
    'test_lu_solve_batched',  # (TPU) precision (1e-6)
    'test_lu_solve',  # (TPU) precision (1e-7)
    'test_solve_batched',  # (TPU) precision (1e-6)
    'test_solve',  # (TPU) precison (1e-7)
    'test_triangular_solve_batched',  # (TPU) precision (1e-6)
    'test_triangular_solve_batched_many_batches',  # (TPU) 1.02 vs 0.001
    'test_triangular_solve',  # (TPU) precision (1e-7)
    'test_scalar_check',  # runtime error
    'test_argminmax_large_axis',  # OOM, and the test is grepping "memory" in the exception message
    'test_trapz', # precision (1e-5), test use np.allClose

    # TestViewOps
    'test_contiguous_nonview',
    'test_expand_as_view',
    'test_expand_view',
    'test_reshape_nonview',
    'test_unfold_view',

    # test_indexing.py
    # TestIndexing
    'test_setitem_expansion_error',  # expecting a different runtime error
    'test_multiple_byte_mask', # expecting a different runtime error
    'test_empty_slice', # stride
    'test_byte_tensor_assignment', # expecting a different runtime error
    'test_byte_mask', # expecting a different runtime error
    'test_byte_mask_accumulate', # expecting a different runtime error
    'test_bool_indices', # expecting a different runtime error
    'test_index_getitem_copy_bools_slices',  # storage
    'test_index_put_byte_indices',  # FIXME: Indexing with uint8 tensor is no longer allowed.
    'test_getitem_scalars',  # storage
    'test_empty_ndim_index',  # expecting a different runtime error

    # NumpyTests
    'test_trivial_fancy_out_of_bounds',  # expecting a different runtime error
    'test_boolean_assignment_value_mismatch',  # expecting a different runtime error
    'test_empty_tuple_index',  # storage
    'test_empty_fancy_index',  # expecting a different runtime error
    'test_ellipsis_index',  # storage
    'test_broaderrors_indexing',  # expecting a different runtime error
    'test_boolean_shape_mismatch',  # expecting a different runtime error
    'test_boolean_indexing_weirdness',  # expecting a different runtime error
    'test_boolean_indexing_weirdness_tensors',  # expecting a different runtime error

    # test_nn.py
    # TestNNDeviceType
    'test_embedding_backward',  # sparse
    'test_embedding_dense_grad',  # slow
    'test_EmbeddingBag_per_sample_weights_and_new_offsets', # FIXME! UndefinedTensorImpl::_singleton
    'test_batchnorm_grad', # FIXME! UndefinedTensorImpl::_singleton
    'test_pool_invalid_size',  # expecting a different runtime error
    'test_nonlinearity_propagate_nan', # relu6 with a nan tensor returns a tensor([0.]) instead of a nan tensor
    'test_InstanceNorm3d_general',  # precision (1e-2)
    'test_InstanceNorm2d_general',  # precision (1e-2)
    'test_InstanceNorm1d_general',  # precision (1e-2)
    'test_EmbeddingBag_per_sample_weights_failures',  # expecting a different runtime error
    'test_variable_sequence',  # PackedSequence batch_sizes.device.type should be CPU but is XLA
    'test_embedding_bag_device',  # FIXME! UndefinedTensorImpl::_singleton
    'test_batchnorm_eval',  # FIXME! UndefinedTensorImpl::_singleton
    'test_MaxPool2d_indices',  # lowering
    'test_MaxPool1d_indices',  # lowering
    'test_EmbeddingBag_per_sample_weights_and_offsets',  # runtime error
    'test_EmbeddingBag_per_sample_weights_and_no_offsets',  # runtime error
    'test_softshrink_negative',  # runtime error
    'test_nll_loss_empty_tensor_reduction_mean',  # floating point division 0 by 0, expecting nan but get 0

    # test_type_promotion.py
    # TestTypePromotion
    'test_many_promotions', # stride
    'test_inplace', # FIXME! XLA allows adding int and double inplace
    'test_indexing', # FIXME! XLA allows int to double type promotion
    'test_alternate_result', # expecting a different runtime error
    'test_half',  # half support
}
