#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
VERBOSITY=2

export TORCH_TEST_DEVICES="$CDIR/pytorch_test_base.py"
export CPU_NUM_DEVICES=4

function run_test {
  coverage run "$@"
}

function run_opbyop {
  echo "Running in OpByOp mode: $@"
  XLA_GET_TENSORS_OPBYOP=1 XLA_SYNC_TENSORS_OPBYOP=1 run_test "$@"
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

function run_xla_backend_mp {
  echo "Running XLA backend multiprocessing test: $@"
  MASTER_ADDR=localhost MASTER_PORT=6000 run_test "$@"
}

function run_pjrt {
  echo "Running in PjRt runtime: $@"
  if [ -x "$(command -v nvidia-smi)" ]; then
    PJRT_DEVICE=GPU run_test "$@"
  else
    # TODO(darisoy): run these tests with multiple CPU devices, this fails due to TF issue.
    PJRT_DEVICE=CPU CPU_NUM_DEVICES=1 run_test "$@"
  fi
}

function run_async_scalar {
  echo "Running in Async Scalar Upload mode: $@"
  XLA_TRANSFER_SCALAR_ASYNC=1 run_test "$@"
}

function run_op_tests {
  COVERAGE_FILE=.coverage_view_ops run_dynamic "$CDIR/../../test/test_view_ops.py" "$@" -v TestViewOpsXLA
  COVERAGE_FILE=.coverage_torch_device_type run_test "$CDIR/../../test/test_torch.py" "$@" -v TestTorchDeviceTypeXLA
  COVERAGE_FILE=.coverage_torch_device_precision run_dynamic "$CDIR/../../test/test_torch.py" "$@" -v TestDevicePrecisionXLA
  COVERAGE_FILE=.coverage_torch_tensor_device_ops run_test "$CDIR/../../test/test_torch.py" "$@" -v TestTensorDeviceOpsXLA
  COVERAGE_FILE=.coverage_indexing run_test "$CDIR/../../test/test_indexing.py" "$@" -v TestIndexingXLA
  COVERAGE_FILE=.coverage_numpy_tests run_test "$CDIR/../../test/test_indexing.py" "$@" -v NumpyTestsXLA
  COVERAGE_FILE=.coverage_nn_device_type run_dynamic "$CDIR/../../test/test_nn.py" "$@" -v TestNNDeviceTypeXLA
  COVERAGE_FILE=.coverage_indexing_dropout_nn_device_type run_dynamic "$CDIR/../../test/nn/test_dropout.py" "$@" -v TestDropoutNNDeviceTypeXLA
  COVERAGE_FILE=.coverage_pooling_device_type run_dynamic "$CDIR/../../test/nn/test_pooling.py" "$@" -v TestPoolingNNDeviceTypeXLA
  COVERAGE_FILE=.coverage_embedding_device_type run_dynamic "$CDIR/../../test/nn/test_embedding.py" "$@" -v TestEmbeddingNNDeviceTypeXLA
  COVERAGE_FILE=.coverage_convolution_nn_device_type run_dynamic "$CDIR/../../test/nn/test_convolution.py" "$@" -v TestConvolutionNNDeviceTypeXLA
  COVERAGE_FILE=.coverage_multihead_attention run_dynamic "$CDIR/../../test/nn/test_multihead_attention.py" "$@" -v TestMultiheadAttentionNNDeviceTypeXLA
  COVERAGE_FILE=.coverage_type_promotion run_dynamic "$CDIR/../../test/test_type_promotion.py" "$@" -v TestTypePromotionXLA
  COVERAGE_FILE=.coverage_operations run_dynamic "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  COVERAGE_FILE=.coverage_dynamic_shapes run_dynamic "$CDIR/test_dynamic_shapes.py" "$@"
  COVERAGE_FILE=.coverage_operations_opbyop run_opbyop "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  COVERAGE_FILE=.coverage_operations_eager run_eager_debug "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  COVERAGE_FILE=.coverage_operations_async_scalar run_async_scalar "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  COVERAGE_FILE=.coverage_operations_grad_checkpoint run_test "$CDIR/test_grad_checkpoint.py"
  COVERAGE_FILE=.coverage_operations_pjrt run_pjrt "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  COVERAGE_FILE=.coverage_async_closures run_test "$CDIR/test_async_closures.py"
  COVERAGE_FILE=.coverage_operationsrun_test "$CDIR/test_xla_dist.py"
  COVERAGE_FILE=.coverage_profiler run_test "$CDIR/test_profiler.py"
  COVERAGE_FILE=.coverage_ops run_test "$CDIR/test_ops.py"
  COVERAGE_FILE=.coverage_metrics run_test "$CDIR/test_metrics.py"
  COVERAGE_FILE=.coverage_dynamo_integrations_util run_test "$CDIR/dynamo/test_dynamo_integrations_util.py"
  COVERAGE_FILE=.coverage_dynamo run_test "$CDIR/dynamo/test_dynamo.py"
  COVERAGE_FILE=.coverage_data_type_downcast_bf16 run_downcast_bf16 "$CDIR/test_data_type.py"
  COVERAGE_FILE=.coverage_data_type_use_bf16 run_use_bf16 "$CDIR/test_data_type.py"
  COVERAGE_FILE=.coverage_distributed_xla_backend run_test "$CDIR/test_torch_distributed_xla_backend.py"
  COVERAGE_FILE=.coverage_env_var_mapper_ir_debug run_xla_ir_debug "$CDIR/test_env_var_mapper.py"
  CCOVERAGE_FILE=.coverage_env_var_mapper_hlo_debug run_xla_hlo_debug "$CDIR/test_env_var_mapper.py"
  COVERAGE_FILE=.coverage_experimental_pjrt run_pjrt "$CDIR/pjrt/test_experimental_pjrt.py"
  COVERAGE_FILE=.coverage_experimental_tpu run_pjrt "$CDIR/pjrt/test_experimental_tpu.py"
  COVERAGE_FILE=.coverage_ddp run_pjrt "$CDIR/pjrt/test_ddp.py"
  COVERAGE_FILE=.coverage_mesh_service run_pjrt "$CDIR/pjrt/test_mesh_service.py"
  COVERAGE_FILE=.coverage_xla_sharding run_pjrt "$CDIR/test_xla_sharding.py"
  COVERAGE_FILE=.coverage_operations_hlo run_test "$CDIR/test_operations_hlo.py" "$@" --verbosity=$VERBOSITY
}

function run_mp_op_tests {
  COVERAGE_FILE=.coverage_mp_replication run_test "$CDIR/test_mp_replication.py"
  COVERAGE_FILE=.coverage_mp_all_to_all run_test "$CDIR/test_mp_all_to_all.py"
  COVERAGE_FILE=.coverage_mp_collective_permute run_test "$CDIR/test_mp_collective_permute.py"
  COVERAGE_FILE=.coverage_mp_all_gather run_test "$CDIR/test_mp_all_gather.py"
  COVERAGE_FILE=.coverage_mp_reduce_scatter run_test "$CDIR/test_mp_reduce_scatter.py"
  COVERAGE_FILE=.coverage_mp_distributed run_test "$CDIR/test_mp_distributed_mm.py"
  COVERAGE_FILE=.coverage_mp_rendezvous run_test "$CDIR/test_mp_rendezvous.py"
  COVERAGE_FILE=.coverage_mp_save run_test "$CDIR/test_mp_save.py"
  COVERAGE_FILE=.coverage_mp_mesh_reduce run_test "$CDIR/test_mp_mesh_reduce.py"
  COVERAGE_FILE=.coverage_mp_sync_batch_norm run_test "$CDIR/test_mp_sync_batch_norm.py"
  COVERAGE_FILE=.coverage_torch_distributed_all_gather_xla_backend run_xla_backend_mp "$CDIR/test_torch_distributed_all_gather_xla_backend.py"
  COVERAGE_FILE=.coverage_torch_distributed_all_reduce_xla_backend run_xla_backend_mp "$CDIR/test_torch_distributed_all_reduce_xla_backend.py"
  COVERAGE_FILE=.coverage_torch_distributed_multi_all_reduce_xla_backend run_xla_backend_mp "$CDIR/test_torch_distributed_multi_all_reduce_xla_backend.py"
  COVERAGE_FILE=.coverage_torch_distributed_reduce_scatter_xla_backend run_xla_backend_mp "$CDIR/test_torch_distributed_reduce_scatter_xla_backend.py"
  COVERAGE_FILE=.coverage_ddp_xla_backend_mp run_xla_backend_mp "$CDIR/test_ddp.py"
}

function run_tests {
  run_op_tests
  run_mp_op_tests
}

run_tests
