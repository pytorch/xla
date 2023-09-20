#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/pytorch_py_test.log
MAX_GRAPH_SIZE=500
GRAPH_CHECK_FREQUENCY=100
VERBOSITY=2

# Note [Keep Going]
#
# Set the `CONTINUE_ON_ERROR` flag to `true` to make the CircleCI tests continue on error.
# This will allow you to see all the failures on your PR, not stopping with the first
# test failure like the default behavior.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  set +e
fi

while getopts 'LM:C:V:' OPTION
do
  case $OPTION in
    L)
      LOGFILE=
      ;;
    M)
      MAX_GRAPH_SIZE=$OPTARG
      ;;
    C)
      GRAPH_CHECK_FREQUENCY=$OPTARG
      ;;
    V)
      VERBOSITY=$OPTARG
      ;;
  esac
done
shift $(($OPTIND - 1))

export TRIM_GRAPH_SIZE=$MAX_GRAPH_SIZE
export TRIM_GRAPH_CHECK_FREQUENCY=$GRAPH_CHECK_FREQUENCY
export TORCH_TEST_DEVICES="$CDIR/pytorch_test_base.py"
export PYTORCH_TEST_WITH_SLOW=1
export XLA_DUMP_FATAL_STACK=1
export CPU_NUM_DEVICES=4

TORCH_XLA_DIR=$(cd ~; dirname "$(python -c 'import torch_xla; print(torch_xla.__file__)')")
COVERAGE_FILE="$CDIR/../.coverage"

function run_coverage {
  if [ "${USE_COVERAGE:-0}" != "0" ]; then
    coverage run --source="$TORCH_XLA_DIR" -p "$@"
  else
    python3 "$@"
  fi
}

function run_test {
  echo "Running in PjRt runtime: $@"
  if [ -x "$(command -v nvidia-smi)" ] && [ "$XLA_CUDA" != "0" ]; then
    PJRT_DEVICE=GPU run_coverage "$@"
  else
    # TODO(darisoy): run these tests with multiple CPU devices, this fails due to TF issue.
    PJRT_DEVICE=CPU CPU_NUM_DEVICES=1 run_coverage "$@"
  fi
}

function run_test_without_functionalization {
  echo "Running with XLA_DISABLE_FUNCTIONALIZATION: $@"
  XLA_DISABLE_FUNCTIONALIZATION=1 run_test "$@"
}

function run_use_bf16 {
  echo "Running with XLA_USE_BF16: $@"
  XLA_USE_BF16=1 run_test "$@"
}

function run_downcast_bf16 {
  echo "Running with XLA_DOWNCAST_BF16: $@"
  XLA_DOWNCAST_BF16=1 run_test "$@"
}

function run_xla_ir_debug {
  echo "Running with XLA_IR_DEBUG: $@"
  XLA_IR_DEBUG=1 run_test "$@"
}

function run_xla_hlo_debug {
  echo "Running with XLA_IR_DEBUG: $@"
  XLA_HLO_DEBUG=1 run_test "$@"
}

function run_dynamic {
  echo "Running in DynamicShape mode: $@"
  XLA_EXPERIMENTAL="nonzero:masked_select:masked_scatter" run_test "$@"
}

function run_eager_debug {
  echo "Running in Eager Debug mode: $@"
  XLA_USE_EAGER_DEBUG_MODE=1 run_test "$@"
}

function run_save_tensor_ir {
  echo "Running in save tensor file mode: $@"
  XLA_SAVE_TENSORS_FILE="/tmp/xla_test_save_ir.txt" XLA_SAVE_TENSORS_FMT="text" run_test "$@"
}

function run_save_tensor_hlo {
  echo "Running in save tensor file mode: $@"
  XLA_SAVE_TENSORS_FILE="/tmp/xla_test_save_ir.txt" XLA_SAVE_TENSORS_FMT="hlo" run_test "$@"
}

function run_stablehlo_compile {
  echo "Running in StableHlo Compile mode: $@"
  XLA_STABLEHLO_COMPILE=1 run_test "$@"
}

function run_xla_backend_mp {
  echo "Running XLA backend multiprocessing test: $@"
  MASTER_ADDR=localhost MASTER_PORT=6000 run_test "$@"
}

function run_torch_op_tests {
  run_dynamic "$CDIR/../../test/test_view_ops.py" "$@" -v TestViewOpsXLA
  run_test_without_functionalization "$CDIR/../../test/test_view_ops.py" "$@" -v TestViewOpsXLA
  run_test "$CDIR/../../test/test_torch.py" "$@" -v TestTorchDeviceTypeXLA
  run_dynamic "$CDIR/../../test/test_torch.py" "$@" -v TestDevicePrecisionXLA
  run_test "$CDIR/../../test/test_torch.py" "$@" -v TestTensorDeviceOpsXLA
  run_test "$CDIR/../../test/test_indexing.py" "$@" -v TestIndexingXLA
  run_test "$CDIR/../../test/test_indexing.py" "$@" -v NumpyTestsXLA
  run_dynamic "$CDIR/../../test/test_nn.py" "$@" -v TestNNDeviceTypeXLA
  run_dynamic "$CDIR/../../test/nn/test_dropout.py" "$@" -v TestDropoutNNDeviceTypeXLA
  run_dynamic "$CDIR/../../test/nn/test_pooling.py" "$@" -v TestPoolingNNDeviceTypeXLA
  run_dynamic "$CDIR/../../test/nn/test_embedding.py" "$@" -v TestEmbeddingNNDeviceTypeXLA
  run_dynamic "$CDIR/../../test/nn/test_convolution.py" "$@" -v TestConvolutionNNDeviceTypeXLA
  run_dynamic "$CDIR/../../test/nn/test_multihead_attention.py" "$@" -v TestMultiheadAttentionNNDeviceTypeXLA
  run_dynamic "$CDIR/../../test/test_type_promotion.py" "$@" -v TestTypePromotionXLA
}

#######################################################################################
################################# XLA OP TESTS SHARDS #################################
#######################################################################################

# DO NOT MODIFY
function run_xla_op_tests1 {
  run_dynamic "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_dynamic "$CDIR/ds/test_dynamic_shapes.py"
  run_dynamic "$CDIR/ds/test_dynamic_shape_models.py" "$@" --verbosity=$VERBOSITY
  run_eager_debug  "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_test "$CDIR/test_grad_checkpoint.py"
  run_test "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_test_without_functionalization "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_test "$CDIR/test_async_closures.py"
  run_test "$CDIR/test_autocast.py"
  run_test "$CDIR/test_profiler.py"
}

# DO NOT MODIFY
function run_xla_op_tests2 {
  run_test "$CDIR/test_ops.py"
  run_test "$CDIR/test_metrics.py"
  run_test "$CDIR/test_zero1.py"
  run_test "$CDIR/dynamo/test_dynamo_integrations_util.py"
  run_test "$CDIR/dynamo/test_dynamo.py"
  run_test "$CDIR/dynamo/test_bridge.py"
  run_test "$CDIR/dynamo/test_num_output.py"
  run_save_tensor_ir "$CDIR/dynamo/test_dynamo_graph_dump.py"
  run_downcast_bf16 "$CDIR/test_data_type.py"
  run_use_bf16 "$CDIR/test_data_type.py"
  run_xla_ir_debug "$CDIR/test_env_var_mapper.py"
  run_xla_hlo_debug "$CDIR/test_env_var_mapper.py"
  run_xla_hlo_debug "$CDIR/stablehlo/test_stablehlo_save_load.py"
  run_save_tensor_ir "$CDIR/spmd/test_spmd_graph_dump.py"
  run_save_tensor_hlo "$CDIR/spmd/test_spmd_graph_dump.py"
  # TODO(qihqi): this test require tensorflow to run. need to setup separate
  #     CI with tf.
  run_xla_hlo_debug "$CDIR/stablehlo/test_stablehlo_inference.py"
  run_stablehlo_compile "$CDIR/stablehlo/test_stablehlo_compile.py"
}

# All the new xla op tests should go to run_xla_op_tests3
function run_xla_op_tests3 {
  run_test "$CDIR/pjrt/test_runtime.py"
  run_test "$CDIR/pjrt/test_runtime_gpu.py"
  run_test "$CDIR/pjrt/test_runtime_multi_cpu.py"
  run_test "$CDIR/pjrt/test_internal_tpu.py"
  run_test "$CDIR/pjrt/test_ddp.py"
  run_test "$CDIR/pjrt/test_mesh_service.py"
  run_test "$CDIR/spmd/test_xla_sharding.py"
  run_test "$CDIR/spmd/test_xla_sharding_hlo.py"
  run_test "$CDIR/spmd/test_xla_virtual_device.py"
  run_test "$CDIR/spmd/test_dynamo_spmd.py"
  run_test "$CDIR/spmd/test_xla_distributed_checkpoint.py"
  run_test "$CDIR/spmd/test_xla_spmd_python_api_interaction.py"
  run_test "$CDIR/test_operations_hlo.py" "$@" --verbosity=$VERBOSITY
  run_test "$CDIR/test_input_output_aliases.py"
  run_test "$CDIR/test_torch_distributed_xla_backend.py"
}

#######################################################################################

function run_op_tests {
  run_torch_op_tests
  run_xla_op_tests1
  run_xla_op_tests2
  run_xla_op_tests3
}

function run_mp_op_tests {
  run_test "$CDIR/test_mp_replication.py"
  run_test "$CDIR/test_mp_all_to_all.py"
  run_test "$CDIR/test_mp_collective_permute.py"
  run_test "$CDIR/test_mp_all_gather.py"
  run_test "$CDIR/test_mp_reduce_scatter.py"
  run_test "$CDIR/test_mp_distributed_mm.py"
  run_test "$CDIR/test_mp_save.py"
  run_test "$CDIR/test_mp_mesh_reduce.py"
  run_test "$CDIR/test_mp_sync_batch_norm.py"
  run_xla_backend_mp "$CDIR/test_torch_distributed_all_gather_xla_backend.py"
  run_xla_backend_mp "$CDIR/test_torch_distributed_all_reduce_xla_backend.py"
  run_xla_backend_mp "$CDIR/test_torch_distributed_multi_all_reduce_xla_backend.py"
  run_xla_backend_mp "$CDIR/test_torch_distributed_reduce_scatter_xla_backend.py"
  run_xla_backend_mp "$CDIR/test_ddp.py"
  run_xla_backend_mp "$CDIR/test_fsdp_auto_wrap.py"
  run_xla_backend_mp "$CDIR/test_torch_distributed_fsdp_meta.py"
}

function run_tests {
  # RUN_ flags filter an explicit test type to run, XLA_SKIP_ flags exclude one.
  if [[ "$RUN_XLA_OP_TESTS1" == "xla_op1" ]]; then
    echo "Running xla op tests..."
    run_xla_op_tests1
  elif [[ "$RUN_XLA_OP_TESTS2" == "xla_op2" ]]; then
    echo "Running xla op tests..."
    run_xla_op_tests2
  elif [[ "$RUN_XLA_OP_TESTS3" == "xla_op3" ]]; then
    echo "Running xla op tests..."
    run_xla_op_tests3
  elif [[ "$RUN_TORCH_OP_TESTS" == "torch_op" ]]; then
    echo "Running torch op tests..."
    run_torch_op_tests
  elif [[ "$RUN_MP_OP_TESTS" == "mp_op" ]]; then
    echo "Running mp op tests..."
    run_mp_op_tests
  else
    # Run full tests without sharding, respects XLA_SKIP_*
    if [[ "$XLA_SKIP_XLA_OP_TESTS" != "1" ]]; then
      run_xla_op_tests1
      run_xla_op_tests2
      run_xla_op_tests3
    fi
    if [[ "$XLA_SKIP_TORCH_OP_TESTS" != "1" ]]; then
      run_torch_op_tests
    fi
    if [[ "$XLA_SKIP_MP_OP_TESTS" != "1" ]]; then
      run_mp_op_tests
    fi
  fi
}

if [ "$LOGFILE" != "" ]; then
  run_tests 2>&1 | tee $LOGFILE
else
  run_tests
fi
