#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")/.." ; pwd -P)"
VERBOSITY=2

export TORCH_TEST_DEVICES="$CDIR/pytorch_test_base.py"
export CPU_NUM_DEVICES=4

cd ~
TORCH_XLA_DIR=$(dirname "$(python -c 'import torch_xla; print(torch_xla.__file__)')")
cd $CDIR

function run_test {
  coverage run --source="$TORCH_XLA_DIR" -p "$@"
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
    # TODO(jonbolin): Only run GPU tests with a single device due to collective failures.
    PJRT_DEVICE=GPU GPU_NUM_DEVICES=1 run_test "$@"
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
  run_dynamic "$CDIR/../../test/test_view_ops.py" "$@" -v TestViewOpsXLA
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
  run_dynamic "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_dynamic "$CDIR/test_dynamic_shapes.py" "$@"
  run_opbyop "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_eager_debug "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_async_scalar "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_test "$CDIR/test_grad_checkpoint.py"
  run_pjrt "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_test "$CDIR/test_async_closures.py"
  run_test "$CDIR/test_xla_dist.py"
  run_test "$CDIR/test_profiler.py"
  run_test "$CDIR/test_ops.py"
  run_test "$CDIR/test_metrics.py"
  run_test "$CDIR/dynamo/test_dynamo_integrations_util.py"
  run_test "$CDIR/dynamo/test_dynamo.py"
  run_downcast_bf16 "$CDIR/test_data_type.py"
  run_use_bf16 "$CDIR/test_data_type.py"
  run_test "$CDIR/test_torch_distributed_xla_backend.py"
  run_xla_ir_debug "$CDIR/test_env_var_mapper.py"
  run_xla_hlo_debug "$CDIR/test_env_var_mapper.py"
  run_pjrt "$CDIR/pjrt/test_experimental_pjrt.py"
  run_pjrt "$CDIR/pjrt/test_experimental_tpu.py"
  run_pjrt "$CDIR/pjrt/test_ddp.py"
  run_pjrt "$CDIR/pjrt/test_mesh_service.py"
  run_pjrt "$CDIR/spmd/test_xla_sharding.py"
  run_test "$CDIR/test_operations_hlo.py" "$@" --verbosity=$VERBOSITY
}

function run_mp_op_tests {
  run_test "$CDIR/test_mp_replication.py"
  run_test "$CDIR/test_mp_all_to_all.py"
  run_test "$CDIR/test_mp_collective_permute.py"
  run_test "$CDIR/test_mp_all_gather.py"
  run_test "$CDIR/test_mp_reduce_scatter.py"
  run_test "$CDIR/test_mp_distributed_mm.py"
  run_test "$CDIR/test_mp_rendezvous.py"
  run_test "$CDIR/test_mp_save.py"
  run_test "$CDIR/test_mp_mesh_reduce.py"
  run_test "$CDIR/test_mp_sync_batch_norm.py"
  run_xla_backend_mp "$CDIR/test_torch_distributed_all_gather_xla_backend.py"
  run_xla_backend_mp "$CDIR/test_torch_distributed_all_reduce_xla_backend.py"
  run_xla_backend_mp "$CDIR/test_torch_distributed_multi_all_reduce_xla_backend.py"
  run_xla_backend_mp "$CDIR/test_torch_distributed_reduce_scatter_xla_backend.py"
  run_xla_backend_mp "$CDIR/test_ddp.py"
}

function run_tests {
  run_op_tests
  run_mp_op_tests
}

run_tests
