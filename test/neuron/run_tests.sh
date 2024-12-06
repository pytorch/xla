#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")"/../ ; pwd -P)"
LOGFILE=/tmp/pytorch_py_test.log
MAX_GRAPH_SIZE=500
GRAPH_CHECK_FREQUENCY=100
VERBOSITY=2

# Note [Keep Going]
#
# Set the `CONTINUE_ON_ERROR` flag to `true` to make the CI tests continue on error.
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
  PJRT_DEVICE=NEURON NEURON_NUM_DEVICES=1 run_coverage "$@"
}

function run_test_without_functionalization {
  echo "Running with XLA_DISABLE_FUNCTIONALIZATION: $@"
  XLA_DISABLE_FUNCTIONALIZATION=1 run_test "$@"
}

function run_xla_ir_debug {
  echo "Running with XLA_IR_DEBUG: $@"
  XLA_IR_DEBUG=1 run_test "$@"
}

function run_use_bf16 {
  echo "Running with XLA_USE_BF16: $@"
  XLA_USE_BF16=1 run_test "$@"
}

function run_downcast_bf16 {
  echo "Running with XLA_DOWNCAST_BF16: $@"
  XLA_DOWNCAST_BF16=1 run_test "$@"
}

function run_xla_hlo_debug {
  echo "Running with XLA_IR_DEBUG: $@"
  XLA_HLO_DEBUG=1 run_test "$@"
}

function run_dynamic {
  echo "Running in DynamicShape mode: $@"
  XLA_EXPERIMENTAL="nonzero:masked_select:masked_scatter:nms" run_test "$@"
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

function run_pt_xla_debug {
  echo "Running in save tensor file mode: $@"
  PT_XLA_DEBUG=1 PT_XLA_DEBUG_FILE="/tmp/pt_xla_debug.txt" run_test "$@"
}

function run_pt_xla_debug_level1 {
  echo "Running in save tensor file mode: $@"
  PT_XLA_DEBUG_LEVEL=1 PT_XLA_DEBUG_FILE="/tmp/pt_xla_debug.txt" run_test "$@"
}

function run_torchrun {
  PJRT_DEVICE=NEURON torchrun --nnodes 1 --nproc-per-node 2 $@
}

function run_torch_op_tests {
  run_dynamic "$CDIR/../../test/test_view_ops.py" "$@" -v TestViewOpsXLA
  run_test_without_functionalization "$CDIR/../../test/test_view_ops.py" "$@" -v TestViewOpsXLA
  run_test "$CDIR/../../test/test_torch.py" "$@" -v TestTorchDeviceTypeXLA
  run_dynamic "$CDIR/../../test/test_torch.py" "$@" -v TestDevicePrecisionXLA
  run_test "$CDIR/../../test/test_torch.py" "$@" -v TestTensorDeviceOpsXLA
  run_test "$CDIR/../../test/test_indexing.py" "$@" -v TestIndexingXLA
  run_test "$CDIR/../../test/test_indexing.py" "$@" -v NumpyTestsXLA
  # run_dynamic "$CDIR/../../test/test_nn.py" "$@" -v TestNNDeviceTypeXLA
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
  #run_dynamic "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  #run_dynamic "$CDIR/ds/test_dynamic_shapes.py"
  #run_dynamic "$CDIR/ds/test_dynamic_shape_models.py" "$@" --verbosity=$VERBOSITY
  #run_eager_debug  "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  #run_test "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  #run_test_without_functionalization "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_pt_xla_debug "$CDIR/debug_tool/test_pt_xla_debug.py"
  run_pt_xla_debug_level1 "$CDIR/debug_tool/test_pt_xla_debug.py"
  run_test "$CDIR/test_async_closures.py"
  run_test "$CDIR/test_hlo_metadata.py"
  #run_test "$CDIR/test_profiler.py"
  run_test "$CDIR/pjrt/test_runtime.py"
  #NEURONCORE_NUM_DEVICES=2 python "$CDIR/pjrt/test_ddp.py"
  run_test "$CDIR/pjrt/test_mesh_service.py"
  #run_test "$CDIR/test_python_ops.py"
  #run_test "$CDIR/test_ops.py"
  run_test "$CDIR/test_metrics.py"
  run_test "$CDIR/test_deprecation.py"
  run_test "$CDIR/dynamo/test_dynamo_integrations_util.py"
  #run_test "$CDIR/dynamo/test_dynamo_aliasing.py"
  run_test "$CDIR/dynamo/test_dynamo.py"
  run_test "$CDIR/dynamo/test_dynamo_dynamic_shape.py"
  run_test "$CDIR/dynamo/test_bridge.py"
  run_test "$CDIR/dynamo/test_num_output.py"
  run_test "$CDIR/dynamo/test_graph_input_matcher.py"
  run_test "$CDIR/dynamo/test_dynamo_config.py"
  run_save_tensor_ir "$CDIR/dynamo/test_dynamo_graph_dump.py"
  #run_test "$CDIR/test_data_type.py"
  run_use_bf16 "$CDIR/test_data_type.py"
  run_downcast_bf16 "$CDIR/test_data_type.py"
  #run_test "$CDIR/test_fp8.py"
  run_xla_ir_debug "$CDIR/test_env_var_mapper.py"
  run_xla_hlo_debug "$CDIR/test_env_var_mapper.py"
  run_xla_hlo_debug "$CDIR/stablehlo/test_stablehlo_save_load.py"
  run_save_tensor_ir "$CDIR/spmd/test_spmd_graph_dump.py"
  run_save_tensor_hlo "$CDIR/spmd/test_spmd_graph_dump.py"
}

function run_xla_op_tests2 {
  run_test "$CDIR/pjrt/test_dtypes.py"
  #run_test "$CDIR/test_while_loop.py"
  run_test "$CDIR/test_scan.py"
  run_test "$CDIR/test_autocast.py"
  run_test "$CDIR/test_grad_checkpoint.py"
  #run_test "$CDIR/eager/test_eager.py"
  run_test "$CDIR/eager/test_eager_with_xla_compile.py"
  run_test "$CDIR/eager/test_eager_with_torch_compile.py"
  #run_test "$CDIR/eager/test_eager_all_reduce_in_place.py"
  run_test "$CDIR/eager/test_eager_spmd.py"
  run_test "$CDIR/test_callback.py"
  XLA_USE_SPMD=1 run_test "$CDIR/test_callback.py"
}

# All the new xla op tests should go to run_xla_op_tests3
function run_xla_op_tests3 {
  # TODO(qihqi): this test require tensorflow to run. need to setup separate
  #     CI with tf.
  run_test "$CDIR/stablehlo/test_exports.py"
  run_test "$CDIR/stablehlo/test_export_fx_passes.py"
  run_test "$CDIR/stablehlo/test_implicit_broadcasting.py"
  run_test "$CDIR/stablehlo/test_composite.py"
  run_test "$CDIR/stablehlo/test_pt2e_qdq.py"
  run_test "$CDIR/stablehlo/test_stablehlo_custom_call.py"
  #run_xla_hlo_debug "$CDIR/stablehlo/test_stablehlo_inference.py"
  #=run_test "$CDIR/stablehlo/test_stablehlo_compile.py"
  run_test "$CDIR/stablehlo/test_unbounded_dynamism.py"
  #run_test "$CDIR/quantized_ops/test_quantized_matmul.py"
  #run_test "$CDIR/quantized_ops/test_dot_general.py"
  #run_test "$CDIR/spmd/test_xla_sharding.py"
  run_test "$CDIR/spmd/test_xla_sharding_hlo.py"
  #run_test "$CDIR/spmd/test_xla_virtual_device.py"
  #run_test "$CDIR/spmd/test_dynamo_spmd.py"
  run_test "$CDIR/spmd/test_spmd_debugging.py"
  #=run_test "$CDIR/spmd/test_xla_distributed_checkpoint.py"
  run_test "$CDIR/spmd/test_xla_spmd_python_api_interaction.py"
  #run_test "$CDIR/spmd/test_dtensor_integration.py"
  #run_test "$CDIR/spmd/test_dtensor_integration2.py"
  run_test "$CDIR/spmd/test_xla_auto_sharding.py"
  #run_test "$CDIR/spmd/test_spmd_parameter_wrapping.py"
  run_test "$CDIR/spmd/test_train_spmd_linear_model.py"
  run_test "$CDIR/spmd/test_xla_spmd_python_api_interaction.py"
  run_test "$CDIR/spmd/test_xla_auto_sharding.py"
  run_test "$CDIR/spmd/test_fsdp_v2.py"
  run_test "$CDIR/test_operations_hlo.py" "$@" --verbosity=$VERBOSITY
  run_test "$CDIR/test_input_output_aliases.py"
  run_test "$CDIR/test_torch_distributed_xla_backend.py"
  run_torchrun "$CDIR/pjrt/test_torchrun.py"
  run_test "$CDIR/test_persistent_cache.py"
  run_test "$CDIR/test_devices.py"

  #python3 examples/data_parallel/train_resnet_xla_ddp.py # compiler error
  #python3 examples/fsdp/train_resnet_fsdp_auto_wrap.py
  #python3 examples/eager/train_decoder_only_eager.py # OOM
  #python3 examples/eager/train_decoder_only_eager_spmd_data_parallel.py # compiler err due to f64
  PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=1 python3 examples/eager/train_decoder_only_eager_with_compile.py # nan loss expected?
  PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=1 python3 examples/eager/train_decoder_only_eager_multi_process.py
}

# Neuron specific tests
function run_xla_neuron_tests {
  run_test "$CDIR/neuron/test_neuron_utils.py"
  run_test "$CDIR/neuron/test_neuron_data_types.py"
}

#######################################################################################

function run_op_tests {
  run_torch_op_tests
  run_xla_op_tests1
  run_xla_op_tests2
  run_xla_op_tests3
  run_xla_neuron_tests
}

function run_mp_op_tests {
  run_test "$CDIR/test_mp_replication.py"
  #run_test "$CDIR/test_mp_all_to_all.py"
  run_test "$CDIR/test_mp_collective_permute.py"
  #run_test "$CDIR/test_mp_all_gather.py"  # "wrong reductions"?
  run_test "$CDIR/test_mp_reduce_scatter.py"
  run_test "$CDIR/test_zero1.py"
  run_test "$CDIR/test_mp_distributed_mm.py"
  run_test "$CDIR/test_mp_save.py"
  run_test "$CDIR/test_mp_mesh_reduce.py"
  run_test "$CDIR/test_mp_sync_batch_norm.py"
  # TODO(JackCaoG): enable this
  run_test "$CDIR/dynamo/test_traceable_collectives.py"
  run_test "$CDIR/test_fsdp_auto_wrap.py"
  # run_torchrun "$CDIR/test_mp_early_exit.py"
  run_pt_xla_debug "$CDIR/debug_tool/test_mp_pt_xla_debug.py"
  run_test "$CDIR/torch_distributed/test_torch_distributed_all_gather_xla_backend.py"
  run_test "$CDIR/torch_distributed/test_torch_distributed_all_reduce_xla_backend.py"
  #run_test "$CDIR/torch_distributed/test_torch_distributed_bucketed_all_reduce_xla_backend.py" # crash without NEURONCORE_NUM_DEVICES=2
  run_test "$CDIR/torch_distributed/test_torch_distributed_multi_all_reduce_xla_backend.py"
  run_test "$CDIR/torch_distributed/test_torch_distributed_reduce_scatter_xla_backend.py"
  run_test "$CDIR/torch_distributed/test_ddp.py"
  #run_test "$CDIR/torch_distributed/test_torch_distributed_fsdp_meta.py" # crash without NEURONCORE_NUM_DEVICES=2
  PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=2 python3 $CDIR/torch_distributed/test_torch_distributed_all_gather_xla_backend.py
  PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=2 python3 $CDIR/torch_distributed/test_torch_distributed_all_reduce_xla_backend.py
  PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=2 python3 $CDIR/torch_distributed/test_torch_distributed_bucketed_all_reduce_xla_backend.py
  PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=2 python3 $CDIR/torch_distributed/test_torch_distributed_multi_all_reduce_xla_backend.py
  PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=2 python3 $CDIR/torch_distributed/test_torch_distributed_reduce_scatter_xla_backend.py
  PJRT_DEVICE=NEURON NEURONCORE_NUM_DEVICES=2 python3 $CDIR/torch_distributed/test_torch_distributed_fsdp_meta.py
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
  elif [[ "$RUN_XLA_NEURON_TESTS" == "xla_neuron" ]]; then
    echo "Running xla neuron tests..."
    run_xla_neuron_tests
  elif [[ "$RUN_TORCH_MP_OP_TESTS" == "torch_mp_op" ]]; then
    echo "Running torch op tests..."
    #run_torch_op_tests
    run_mp_op_tests
  else
    # Run full tests without sharding, respects XLA_SKIP_*
    if [[ "$XLA_SKIP_XLA_OP_TESTS" != "1" ]]; then
      run_xla_op_tests1
      run_xla_op_tests2
      run_xla_op_tests3
    fi
    #if [[ "$XLA_SKIP_TORCH_OP_TESTS" != "1" ]]; then
    #  run_torch_op_tests
    #fi
    if [[ "$XLA_SKIP_MP_OP_TESTS" != "1" ]]; then
      run_mp_op_tests
    fi
    if [[ "$XLA_SKIP_NEURON_TESTS" != "1" ]]; then
      run_xla_neuron_tests
    fi
  fi
}

if [ "$LOGFILE" != "" ]; then
  run_tests 2>&1 | tee $LOGFILE
else
  run_tests
fi
