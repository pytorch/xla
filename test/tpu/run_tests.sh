#!/bin/bash
set -xue

# Absolute path to the directory of this script.
_TPU_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"

# Absolute path to the test/ directory.
_TEST_DIR="$(dirname "$_TPU_DIR")"
_TORCH_XLA_DIR=$(cd ~; dirname "$(python -c 'import torch_xla; print(torch_xla.__file__)')")

source "${_TEST_DIR}/utils/run_tests_utils.sh"

while getopts 'h' OPTION
do
case $OPTION in
    h)
    echo -e "Usage: $0 TEST_FILTER...\nwhere TEST_FILTERs are globs match .py test files. If no test filter is provided, runs all tests."
    exit 0
    ;;
    \?)  # This catches all invalid options.
    echo "ERROR: Invalid commandline flag."
    exit 1
esac
done

# Consume the parsed commandline arguments.
shift $(($OPTIND - 1))

set_test_filter $@

function run_test {
  if ! test_is_selected "$1"; then
    return
  fi
  run_coverage "$@"
}

function run_coverage {
  if [ "${USE_COVERAGE:-0}" != "0" ]; then
    coverage run --source="$_TORCH_XLA_DIR" -p "$@"
  else
    python3 "$@"
  fi
}

function run {
  # TODO: merge with other run_tests
  if test_is_selected $_TEST_DIR/test_mat_mul_precision.py; then
    (cd $_TEST_DIR && run_coverage -m unittest test_mat_mul_precision.TestMatMulPrecision.test_high)
    (cd $_TEST_DIR && run_coverage -m unittest test_mat_mul_precision.TestMatMulPrecision.test_default)
    (cd $_TEST_DIR && run_coverage -m unittest test_mat_mul_precision.TestMatMulPrecision.test_highest)
    (cd $_TEST_DIR && run_coverage -m unittest test_mat_mul_precision.TestMatMulPrecision.test_all)
  fi
  run_coverage "$TEST_CDIR/test_mat_mul_precision_get_and_set.py"
  run_coverage "$TEST_CDIR/test_operations.py" -v
  run_coverage "$TEST_CDIR/pjrt/test_runtime_tpu.py"
  run_coverage "$TEST_CDIR/pjrt/test_collective_ops_tpu.py"
  run_coverage "$TEST_CDIR/spmd/test_mp_input_sharding.py"
  run_coverage "$TEST_CDIR/test_mp_collective_matmul.py"
  run_save_tensor_hlo run_coverage "$TEST_CDIR/spmd/test_spmd_lowering_context.py"
  run_coverage "$TEST_CDIR/spmd/test_xla_sharding.py"
  run_coverage "$TEST_CDIR/spmd/test_xla_virtual_device.py"
  run_coverage "$TEST_CDIR/spmd/test_xla_distributed_checkpoint.py"
  run_coverage "$TEST_CDIR/spmd/test_train_spmd_linear_model.py"
  run_coverage "$TEST_CDIR/spmd/test_xla_spmd_python_api_interaction.py"
  run_coverage "$TEST_CDIR/spmd/test_xla_auto_sharding.py"
  run_coverage "$TEST_CDIR/spmd/test_fsdp_v2.py"
  run_coverage "$TEST_CDIR/test_gradient_accumulation.py"
  XLA_EXPERIMENTAL=nonzero:masked_select:nms run_coverage "$TEST_CDIR/ds/test_dynamic_shape_models.py" -v
  run_coverage "$TEST_CDIR/test_autocast.py"
  run_coverage "$TEST_CDIR/test_fp8.py"
  run_coverage "$TEST_CDIR/test_grad_checkpoint.py"
  run_coverage "$TEST_CDIR/test_grad_checkpoint.py" "$@" --test_autocast
  run_coverage "$TEST_CDIR/dynamo/test_dynamo.py"
  run_coverage "$TEST_CDIR/dynamo/test_dynamo_dynamic_shape.py"
  run_coverage "$TEST_CDIR/spmd/test_spmd_debugging.py"
  XLA_PARAMETER_WRAPPING_THREADSHOLD=1 run_coverage "$TEST_CDIR/spmd/test_spmd_parameter_wrapping.py"
  run_coverage "$TEST_CDIR/pjrt/test_dtypes.py"
  run_coverage "$TEST_CDIR/pjrt/test_dynamic_plugin_tpu.py"
  run_coverage "$TEST_CDIR/test_while_loop.py"
  run_coverage "$TEST_CDIR/scan/test_scan.py"
  run_coverage "$TEST_CDIR/scan/test_scan_spmd.py"
  run_coverage "$TEST_CDIR/scan/test_scan_pallas.py"
  run_coverage "$TEST_CDIR/scan/test_scan_layers.py"
  run_coverage "$TEST_CDIR/test_gru.py"
  run_coverage "$TEST_CDIR/test_assume_pure.py"
  run_coverage "$TEST_CDIR/test_assume_pure_spmd.py"
  run_coverage "$TEST_CDIR/test_as_stride_use_slice.py"
  run_xla_hlo_debug run_coverage "$TEST_CDIR/scan/test_scan_debug.py"
  run_coverage "$TEST_CDIR/test_pallas.py" -v
  run_coverage "$TEST_CDIR/test_pallas_spmd.py"
  XLA_DISABLE_FUNCTIONALIZATION=1 run_coverage "$TEST_CDIR/test_pallas_spmd.py"
  run_coverage "$TEST_CDIR/test_splash_attention.py"
  run_coverage "$TEST_CDIR/test_profiler_session.py"
  run_coverage "$TEST_CDIR/test_multi_queries_paged_attention_kernel.py"
  run_coverage "$TEST_CDIR/test_ragged_paged_attention_kernel.py"
  run_coverage "$TEST_CDIR/test_input_output_aliases.py"
  run_coverage "$TEST_CDIR/test_gmm.py"
  run_coverage "$TEST_CDIR/eager/test_eager_spmd.py"
  run_coverage "$TEST_CDIR/torch_distributed/test_torch_distributed_all_gather_xla_backend.py"
  run_coverage "$TEST_CDIR/torch_distributed/test_torch_distributed_all_reduce_xla_backend.py"
  run_coverage "$TEST_CDIR/torch_distributed/test_torch_distributed_multi_all_reduce_xla_backend.py"
  run_coverage "$TEST_CDIR/torch_distributed/test_torch_distributed_reduce_scatter_xla_backend.py"
  run_coverage "$TEST_CDIR/quantized_ops/test_dot_general.py"
  run_xla_ir_hlo_debug run_coverage "$TEST_CDIR/test_user_computation_debug_cache.py"
  run_coverage "$TEST_CDIR/test_data_type.py"
  run_coverage "$TEST_CDIR/test_compilation_cache_utils.py"
  
  # run examples, each test should takes <2 minutes
  run_coverage "$TEST_CDIR/../examples/data_parallel/train_resnet_spmd_data_parallel.py"
  run_coverage "$TEST_CDIR/../examples/fsdp/train_decoder_only_fsdp_v2.py"
  run_coverage "$TEST_CDIR/../examples/train_resnet_amp.py"
  run_coverage "$TEST_CDIR/../examples/train_decoder_only_base.py"
  run_coverage "$TEST_CDIR/../examples/train_decoder_only_base.py" scan.decoder_with_scan.DecoderWithScan \
      --num-steps 30  # TODO(https://github.com/pytorch/xla/issues/8632): Reduce scan tracing overhead
  
  # HACK: don't confuse local `torch_xla` folder with installed package
  # Python 3.11 has the permanent fix: https://stackoverflow.com/a/73636559
  # Egaer tests will take more HBM, only run them on TPU v4 CI
  TPU_VERSION=$(python -c "import sys; sys.path.remove(''); import torch_xla; print(torch_xla._internal.tpu.version())")
  if [[ -n "$TPU_VERSION" && "$TPU_VERSION" == "4" ]]; then
      run_coverage "$TEST_CDIR/dynamo/test_traceable_collectives.py"
      run_coverage "$TEST_CDIR/../examples/data_parallel/train_resnet_xla_ddp.py"
      run_coverage "$TEST_CDIR/../examples/fsdp/train_resnet_fsdp_auto_wrap.py"
      run_coverage "$TEST_CDIR/../examples/eager/train_decoder_only_eager.py"
      run_coverage "$TEST_CDIR/../examples/eager/train_decoder_only_eager_spmd_data_parallel.py"
      run_coverage "$TEST_CDIR/../examples/eager/train_decoder_only_eager_with_compile.py"
      run_coverage "$TEST_CDIR/../examples/eager/train_decoder_only_eager_multi_process.py"
      XLA_EXPERIMENTAL=nonzero:masked_select:nms run_coverage "$TEST_CDIR/ds/test_dynamic_shapes.py" -v
  fi
  
  if [[ -n "$TPU_VERSION" && "$TPU_VERSION" != "6" ]]; then
      # Test `tpu-info` CLI compatibility
      run_coverage "$CDIR/tpu_info/test_cli.py"
  fi
}

if [ "$USE_COVERAGE" != "0" ]; then
  run
  coverage combine
  coverage-lcov --data_file_path $COVERAGE_FILE --output_file_path $COVERAGE_FILE.info
else
  run
fi
