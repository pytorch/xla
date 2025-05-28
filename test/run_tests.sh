#!/bin/bash
set -exo pipefail

# Absolute path to the directory of this script.
_TEST_DIR="$(
  cd "$(dirname "$0")"
  pwd -P
)"

# Import utilities.
source "${_TEST_DIR}/utils/run_tests_utils.sh"

parse_options_to_vars $@

# Consume the parsed commandline arguments.
shift $(($OPTIND - 1))

# Set the `CONTINUE_ON_ERROR` flag to `1` to make the CI tests continue on error.
# This will allow you to see all the failures on your PR, not stopping with the first
# test failure like the default behavior.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  set +e
fi

export TORCH_TEST_DEVICES="$_TEST_DIR/pytorch_test_base.py"
export PYTORCH_TEST_WITH_SLOW=1
export XLA_DUMP_FATAL_STACK=1
export CPU_NUM_DEVICES=4

_TORCH_XLA_DIR=$(
  cd ~
  dirname "$(python -c 'import torch_xla; print(torch_xla.__file__)')"
)
COVERAGE_FILE="$_TEST_DIR/../.coverage"

function run_coverage {
  if ! test_is_selected "$1"; then
    return
  fi
  if [ "${USE_COVERAGE:-0}" != "0" ]; then
    coverage run --source="$_TORCH_XLA_DIR" -p "$@"
  else
    python3 "$@"
  fi
}

function run_test {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running in PjRt runtime: $@"
  if [ -x "$(command -v nvidia-smi)" ] && [ "$XLA_CUDA" != "0" ]; then
    PJRT_DEVICE=CUDA run_coverage "$@"
  else
    # TODO(darisoy): run these tests with multiple CPU devices, this fails due to TF issue.
    PJRT_DEVICE=CPU CPU_NUM_DEVICES=1 run_coverage "$@"
  fi
}

function run_device_detection_test {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running in PjRt runtime: $@"
  current_device=$PJRT_DEVICE
  current_num_gpu_devices=$GPU_NUM_DEVICES

  unset PJRT_DEVICE
  unset GPU_NUM_DEVICES

  run_coverage $@

  export GPU_NUM_DEVICES=$current_num_gpu_devices
  export PJRT_DEVICE=$current_device
}

function run_test_without_functionalization {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running with XLA_DISABLE_FUNCTIONALIZATION: $@"
  XLA_DISABLE_FUNCTIONALIZATION=1 run_test "$@"
}

function run_use_bf16 {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running with XLA_USE_BF16: $@"
  XLA_USE_BF16=1 run_test "$@"
}

function run_downcast_bf16 {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running with XLA_DOWNCAST_BF16: $@"
  XLA_DOWNCAST_BF16=1 run_test "$@"
}

function run_dynamic {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running in DynamicShape mode: $@"
  XLA_EXPERIMENTAL="nonzero:masked_select:masked_scatter:nms" run_test "$@"
}

function run_eager_debug {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running in Eager Debug mode: $@"
  XLA_USE_EAGER_DEBUG_MODE=1 run_test "$@"
}

function run_pt_xla_debug {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running in save tensor file mode: $@"
  PT_XLA_DEBUG=1 PT_XLA_DEBUG_FILE="/tmp/pt_xla_debug.txt" run_test "$@"
}

function run_pt_xla_debug_level1 {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running in save tensor file mode: $@"
  PT_XLA_DEBUG_LEVEL=1 PT_XLA_DEBUG_FILE="/tmp/pt_xla_debug.txt" run_test "$@"
}

function run_pt_xla_debug_level2 {
  if ! test_is_selected "$1"; then
    return
  fi
  echo "Running in save tensor file mode: $@"
  PT_XLA_DEBUG_LEVEL=2 PT_XLA_DEBUG_FILE="/tmp/pt_xla_debug.txt" run_test "$@"
}

function run_torchrun {
  if ! test_is_selected "$1"; then
    return
  fi
  if [ -x "$(command -v nvidia-smi)" ] && [ "$XLA_CUDA" != "0" ]; then
    echo "Running torchrun test for GPU $@"
    num_devices=$(nvidia-smi --list-gpus | wc -l)
    PJRT_DEVICE=CUDA torchrun --nnodes 1 --nproc-per-node $num_devices $@
  fi
}

function run_torch_op_tests {
  run_dynamic "$_TEST_DIR/../../test/test_view_ops.py" "$@" -v TestViewOpsXLA
  run_test_without_functionalization "$_TEST_DIR/../../test/test_view_ops.py" "$@" -v TestViewOpsXLA
  run_test "$_TEST_DIR/../../test/test_torch.py" "$@" -v TestTorchDeviceTypeXLA
  run_dynamic "$_TEST_DIR/../../test/test_torch.py" "$@" -v TestDevicePrecisionXLA
  run_test "$_TEST_DIR/../../test/test_torch.py" "$@" -v TestTensorDeviceOpsXLA
  run_test "$_TEST_DIR/../../test/test_indexing.py" "$@" -v TestIndexingXLA
  run_test "$_TEST_DIR/../../test/test_indexing.py" "$@" -v NumpyTestsXLA
  # run_dynamic "$_TEST_DIR/../../test/test_nn.py" "$@" -v TestNNDeviceTypeXLA
  run_dynamic "$_TEST_DIR/../../test/nn/test_dropout.py" "$@" -v TestDropoutNNDeviceTypeXLA
  run_dynamic "$_TEST_DIR/../../test/nn/test_pooling.py" "$@" -v TestPoolingNNDeviceTypeXLA
  run_dynamic "$_TEST_DIR/../../test/nn/test_embedding.py" "$@" -v TestEmbeddingNNDeviceTypeXLA
  run_dynamic "$_TEST_DIR/../../test/nn/test_convolution.py" "$@" -v TestConvolutionNNDeviceTypeXLA
  run_dynamic "$_TEST_DIR/../../test/nn/test_multihead_attention.py" "$@" -v TestMultiheadAttentionNNDeviceTypeXLA
  run_dynamic "$_TEST_DIR/../../test/test_type_promotion.py" "$@" -v TestTypePromotionXLA
}

#######################################################################################
################################# XLA OP TESTS SHARDS #################################
#######################################################################################

# DO NOT MODIFY
function run_xla_op_tests1 {
  run_dynamic "$_TEST_DIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_dynamic "$_TEST_DIR/ds/test_dynamic_shapes.py"
  run_dynamic "$_TEST_DIR/ds/test_dynamic_shape_models.py" "$@" --verbosity=$VERBOSITY
  run_eager_debug "$_TEST_DIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_test "$_TEST_DIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_test "$_TEST_DIR/test_xla_graph_execution.py" "$@" --verbosity=$VERBOSITY
  run_pt_xla_debug_level2 "$_TEST_DIR/test_xla_graph_execution.py" "$@" --verbosity=$VERBOSITY
  run_test_without_functionalization "$_TEST_DIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_pt_xla_debug "$_TEST_DIR/debug_tool/test_pt_xla_debug.py"
  run_pt_xla_debug_level1 "$_TEST_DIR/debug_tool/test_pt_xla_debug.py"
  run_test "$_TEST_DIR/test_async_closures.py"
  run_test "$_TEST_DIR/test_hlo_metadata.py"
  # TODO(https://github.com/pytorch/xla/issues/8796): Re-enable this test
  # run_test "$_TEST_DIR/test_profiler.py"
  run_test "$_TEST_DIR/test_profiler_session.py"
  run_test "$_TEST_DIR/pjrt/test_runtime.py"
  run_test "$_TEST_DIR/pjrt/test_runtime_single_proc_gpu.py"

  # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
  # Re-enable this test on GPU.
  # run_test "$_TEST_DIR/pjrt/test_runtime_multi_gpu.py"

  run_test "$_TEST_DIR/pjrt/test_runtime_multi_cpu.py"
  run_test "$_TEST_DIR/pjrt/test_internal_tpu.py"

  # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
  # Re-enable this test on GPU.
  PJRT_DEVICE=CPU XLA_CUDA=0 run_test "$_TEST_DIR/pjrt/test_ddp.py"

  # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
  # Re-enable this test on GPU.
  PJRT_DEVICE=CPU XLA_CUDA=0 run_test "$_TEST_DIR/pjrt/test_mesh_service.py"

  run_test "$_TEST_DIR/test_python_ops.py"
  run_test "$_TEST_DIR/test_ops.py"
  run_test "$_TEST_DIR/test_metrics.py"
  if [ -f "/tmp/metrics.txt" ]; then
    rm /tmp/metrics.txt
  fi
  XLA_METRICS_FILE=/tmp/metrics.txt run_test "$_TEST_DIR/test_metrics.py"
  run_test "$_TEST_DIR/test_deprecation.py"
  run_test "$_TEST_DIR/dynamo/test_dynamo_integrations_util.py"
  run_test "$_TEST_DIR/dynamo/test_dynamo_aliasing.py"
  run_test "$_TEST_DIR/dynamo/test_dynamo.py"
  run_test "$_TEST_DIR/dynamo/test_dynamo_dynamic_shape.py"
  run_test "$_TEST_DIR/dynamo/test_bridge.py"
  run_test "$_TEST_DIR/dynamo/test_num_output.py"
  run_test "$_TEST_DIR/dynamo/test_graph_input_matcher.py"
  run_test "$_TEST_DIR/dynamo/test_dynamo_config.py"
  run_save_tensor_ir run_test "$_TEST_DIR/dynamo/test_dynamo_graph_dump.py"
  run_test "$_TEST_DIR/test_data_type.py"
  run_test "$_TEST_DIR/test_fp8.py"
  run_xla_ir_debug run_test "$_TEST_DIR/test_env_var_mapper.py"
  run_xla_hlo_debug run_test "$_TEST_DIR/test_env_var_mapper.py"
  run_xla_hlo_debug run_test "$_TEST_DIR/stablehlo/test_stablehlo_save_load.py"
  run_save_tensor_ir run_test "$_TEST_DIR/spmd/test_spmd_graph_dump.py"
  run_save_tensor_hlo run_test "$_TEST_DIR/spmd/test_spmd_graph_dump.py"
}

function run_xla_op_tests2 {
  run_test "$_TEST_DIR/pjrt/test_dtypes.py"
  run_test "$_TEST_DIR/test_while_loop.py"
  run_test "$_TEST_DIR/scan/test_scan.py"
  run_test "$_TEST_DIR/scan/test_scan_spmd.py"
  run_test "$_TEST_DIR/scan/test_scan_layers.py"
  run_test "$_TEST_DIR/test_gru.py"
  run_test "$_TEST_DIR/test_as_stride_use_slice.py"
  run_test "$_TEST_DIR/test_placeholder.py"
  run_xla_hlo_debug run_test "$_TEST_DIR/scan/test_scan_debug.py"
  run_test "$_TEST_DIR/test_autocast.py"
  run_test "$_TEST_DIR/eager/test_eager.py"
  run_test "$_TEST_DIR/eager/test_eager_with_xla_compile.py"
  run_test "$_TEST_DIR/eager/test_eager_with_torch_compile.py"

  # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
  # Re-enable this test on GPU.
  PJRT_DEVICE=CPU XLA_CUDA=0 run_test "$_TEST_DIR/eager/test_eager_all_reduce_in_place.py"

  run_test "$_TEST_DIR/eager/test_eager_spmd.py"
  run_test "$_TEST_DIR/test_callback.py"
  XLA_USE_SPMD=1 run_test "$_TEST_DIR/test_callback.py"
  run_test "$_TEST_DIR/test_jax_interop.py"
  run_test "$_TEST_DIR/test_assume_pure.py"
  run_test "$_TEST_DIR/test_assume_pure_spmd.py"
  run_test "$_TEST_DIR/test_assume_pure_torch.py"
  run_test "$_TEST_DIR/test_dynamic_shapes_detector.py"
}

# All the new xla op tests should go to run_xla_op_tests3
function run_xla_op_tests3 {
  # TODO(qihqi): this test require tensorflow to run. need to setup separate
  #     CI with tf.
  run_test "$_TEST_DIR/stablehlo/test_exports.py"
  run_test "$_TEST_DIR/stablehlo/test_export_fx_passes.py"
  run_test "$_TEST_DIR/stablehlo/test_implicit_broadcasting.py"
  run_test "$_TEST_DIR/stablehlo/test_composite.py"
  run_test "$_TEST_DIR/stablehlo/test_pt2e_qdq.py"
  run_test "$_TEST_DIR/stablehlo/test_stablehlo_custom_call.py"
  run_xla_hlo_debug run_test "$_TEST_DIR/stablehlo/test_stablehlo_inference.py"
  run_test "$_TEST_DIR/stablehlo/test_stablehlo_compile.py"
  run_test "$_TEST_DIR/stablehlo/test_unbounded_dynamism.py"
  run_test "$_TEST_DIR/quantized_ops/test_quantized_matmul.py"
  run_test "$_TEST_DIR/quantized_ops/test_dot_general.py"
  run_test "$_TEST_DIR/spmd/test_xla_sharding.py"
  run_test "$_TEST_DIR/spmd/test_xla_sharding_hlo.py"
  run_test "$_TEST_DIR/spmd/test_xla_virtual_device.py"
  run_test "$_TEST_DIR/spmd/test_dynamo_spmd.py"
  run_test "$_TEST_DIR/spmd/test_spmd_debugging.py"
  run_test "$_TEST_DIR/spmd/test_xla_distributed_checkpoint.py"
  run_test "$_TEST_DIR/spmd/test_xla_spmd_python_api_interaction.py"
  run_test "$_TEST_DIR/spmd/test_dtensor_integration.py"
  run_test "$_TEST_DIR/spmd/test_dtensor_integration2.py"
  run_test "$_TEST_DIR/spmd/test_xla_auto_sharding.py"
  run_test "$_TEST_DIR/spmd/test_spmd_parameter_wrapping.py"
  run_test "$_TEST_DIR/spmd/test_mp_input_sharding.py"
  run_test "$_TEST_DIR/spmd/test_train_spmd_linear_model.py" "$@" --skip-gradient-checkpointing
  run_test "$_TEST_DIR/test_gradient_accumulation.py"
  run_save_tensor_hlo run_test "$_TEST_DIR/spmd/test_spmd_lowering_context.py"
  run_test "$_TEST_DIR/test_operations_hlo.py" "$@" --verbosity=$VERBOSITY
  run_test "$_TEST_DIR/test_input_output_aliases.py"
  run_test_without_functionalization "$_TEST_DIR/test_input_output_aliases.py"
  run_test "$_TEST_DIR/test_torch_distributed_xla_backend.py"

  # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
  # Re-enable this test on GPU.
  PJRT_DEVICE=CPU XLA_CUDA=0 run_torchrun "$_TEST_DIR/pjrt/test_torchrun.py"

  run_test "$_TEST_DIR/test_compilation_cache_utils.py"
  run_test "$_TEST_DIR/test_persistent_cache.py"
  run_test "$_TEST_DIR/test_devices.py"
  run_device_detection_test "$_TEST_DIR/test_gpu_device_detection.py"
  run_test "$_TEST_DIR/test_manual_xla_registration.py"
  # NOTE: this line below is testing export and don't care about GPU
  PJRT_DEVICE=CPU CPU_NUM_DEVICES=1 run_coverage "$_TEST_DIR/test_core_aten_ops.py"
  run_test "$_TEST_DIR/test_pallas.py"
  run_xla_ir_hlo_debug run_test "$_TEST_DIR/test_user_computation_debug_cache.py"

  # Test examples
  run_test "$_TEST_DIR/../examples/scan/scan_examples.py"

  # CUDA tests
  if [ -x "$(command -v nvidia-smi)" ]; then
    # Please keep PJRT_DEVICE and GPU_NUM_DEVICES explicit in the following test commands.
    echo "single-host-single-process"
    PJRT_DEVICE=CUDA GPU_NUM_DEVICES=1 python3 test/test_train_mp_imagenet.py --fake_data --batch_size=16 --num_epochs=1 --num_cores=1 --num_steps=25 --model=resnet18
    PJRT_DEVICE=CUDA torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 test/test_train_mp_imagenet.py --fake_data --pjrt_distributed --batch_size=16 --num_epochs=1 --num_steps=25 --model=resnet18

    echo "single-host-multi-process"
    num_devices=$(nvidia-smi --list-gpus | wc -l)
    # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
    # Re-enable this test on GPU.
    # PJRT_DEVICE=CUDA GPU_NUM_DEVICES=$num_devices python3 test/test_train_mp_imagenet.py --fake_data --batch_size=16 --num_epochs=1 --num_steps=25 --model=resnet18
    # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
    # Re-enable this test on GPU.
    # PJRT_DEVICE=CUDA torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$num_devices test/test_train_mp_imagenet.py --fake_data --pjrt_distributed --batch_size=16 --num_epochs=1  --num_steps=25 --model=resnet18

    echo "single-host-SPMD"
    # TODO: Reduce BS due to GPU test OOM in CI after pin update to 03/05/2024 (#6677)
    XLA_USE_SPMD=1 PJRT_DEVICE=CUDA torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 test/spmd/test_train_spmd_imagenet.py --fake_data --batch_size 8 --sharding=batch --num_epochs=1 --num_steps=25 --model=resnet18

    # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
    # Re-enable this test on GPU.
    # TODO: Reduce BS due to GPU test OOM in CI after pin update to 03/05/2024 (#6677)
    # PJRT_DEVICE=CUDA python test/test_train_mp_imagenet_fsdp.py --fake_data --use_nested_fsdp --use_small_fake_sample --num_epochs=1 --batch_size 32 --test_set_batch_size 32
    # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
    # Re-enable this test on GPU.
    # TODO: Reduce BS due to GPU test OOM in CI after pin update to 03/05/2024 (#6677)
    # PJRT_DEVICE=CUDA python test/test_train_mp_imagenet_fsdp.py --fake_data --auto_wrap_policy type_based --use_small_fake_sample --num_epochs=1 --batch_size 32 --test_set_batch_size 32
    # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
    # Re-enable this test on GPU.
    # TODO: Reduce BS due to GPU test OOM in CI after pin update to 03/05/2024 (#6677)
    # XLA_DISABLE_FUNCTIONALIZATION=1 PJRT_DEVICE=CUDA python test/test_train_mp_imagenet_fsdp.py --fake_data --use_nested_fsdp --use_small_fake_sample --num_epochs=1

    # Syncfree SGD optimizer tests
    if [ -d ./torch_xla/amp/syncfree ]; then
      echo "Running Syncfree Optimizer Test"
      PJRT_DEVICE=CUDA python test/test_syncfree_optimizers.py

      # Following test scripts are mainly useful for
      # performance evaluation & comparison among different
      # amp optimizers.
      # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
      # Re-enable this test on GPU.
      # echo "Running ImageNet Test"
      # PJRT_DEVICE=CUDA GPU_NUM_DEVICES=$num_devices python test/test_train_mp_imagenet_amp.py --fake_data --num_epochs=1 --batch_size 64 --num_steps=25 --model=resnet18

      # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
      # Re-enable this test on GPU.
      # echo "Running MNIST Test"
      # PJRT_DEVICE=CUDA GPU_NUM_DEVICES=$num_devices python test/test_train_mp_mnist_amp.py --fake_data --num_epochs=1 --batch_size 64 --num_steps=25
    fi
  fi
}

#######################################################################################

function run_op_tests {
  run_torch_op_tests
  run_xla_op_tests1
  run_xla_op_tests2
  run_xla_op_tests3
}

function run_mp_op_tests {
  run_test "$_TEST_DIR/test_mp_replication.py"
  run_test "$_TEST_DIR/test_mp_all_to_all.py"
  run_test "$_TEST_DIR/test_mp_all_gather.py"
  run_test "$_TEST_DIR/test_mp_reduce_scatter.py"
  run_test "$_TEST_DIR/test_zero1.py"
  run_test "$_TEST_DIR/test_mp_distributed_mm.py"
  run_test "$_TEST_DIR/test_mp_save.py"
  run_test "$_TEST_DIR/test_mp_mesh_reduce.py"
  run_test "$_TEST_DIR/test_mp_sync_batch_norm.py"
  # TODO(JackCaoG): enable this
  run_test "$_TEST_DIR/dynamo/test_traceable_collectives.py"
  run_test "$_TEST_DIR/test_fsdp_auto_wrap.py"
  # run_torchrun "$_TEST_DIR/test_mp_early_exit.py"
  run_pt_xla_debug "$_TEST_DIR/debug_tool/test_mp_pt_xla_debug.py"
  run_test "$_TEST_DIR/torch_distributed/test_torch_distributed_all_gather_xla_backend.py"
  run_test "$_TEST_DIR/torch_distributed/test_torch_distributed_all_reduce_xla_backend.py"
  run_test "$_TEST_DIR/torch_distributed/test_torch_distributed_bucketed_all_reduce_xla_backend.py"
  run_test "$_TEST_DIR/torch_distributed/test_torch_distributed_multi_all_reduce_xla_backend.py"
  run_test "$_TEST_DIR/torch_distributed/test_torch_distributed_reduce_scatter_xla_backend.py"
  run_test "$_TEST_DIR/torch_distributed/test_ddp.py"
  run_test "$_TEST_DIR/torch_distributed/test_torch_distributed_fsdp_meta.py"
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
  elif [[ "$RUN_TORCH_MP_OP_TESTS" == "torch_mp_op" ]]; then
    echo "Running torch op tests..."
    run_torch_op_tests

    # TODO(https://github.com/pytorch/xla/issues/9066): GPU testing disabled to unblock pin update.
    # Re-enable this test on GPU.
    PJRT_DEVICE=CPU XLA_CUDA=0 run_mp_op_tests
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

set_test_filter $@

if [ "$LOGFILE" != "" ]; then
  run_tests 2>&1 | tee $LOGFILE
else
  run_tests
fi
