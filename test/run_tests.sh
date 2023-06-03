#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/pytorch_py_test.log
MAX_GRAPH_SIZE=500
GRAPH_CHECK_FREQUENCY=100
VERBOSITY=2

BAZEL_REMOTE_CACHE="0"
BAZEL_VERB="test"

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
    R)
      BAZEL_REMOTE_CACHE="1"
      ;;
    C)
      BAZEL_VERB="coverage"
      ;;
  esac
done
shift $(($OPTIND - 1))

if [ "${USE_COVERAGE:-0}" != "0" ]; then
  BAZEL_VERB="coverage"
fi

export TRIM_GRAPH_SIZE=$MAX_GRAPH_SIZE
export TRIM_GRAPH_CHECK_FREQUENCY=$GRAPH_CHECK_FREQUENCY
export TORCH_TEST_DEVICES="$CDIR/pytorch_test_base.py"
export PYTORCH_TEST_WITH_SLOW=1
export XLA_DUMP_FATAL_STACK=1
#export CPU_NUM_DEVICES=4

TORCH_XLA_DIR=$(cd ~; dirname "$(python -c 'import torch_xla; print(torch_xla.__file__)')")
COVERAGE_FILE="$CDIR/../.coverage"

function run_coverage {
  if [ "${USE_COVERAGE:-0}" != "0" ]; then
    coverage run --source="$TORCH_XLA_DIR" -p "$@"
  else
    python3 "$@"
  fi
}

function set_accelerator {
  if [ -x "$(command -v nvidia-smi)" ] && [ "$XLA_CUDA" != "0" ]; then
    BAZEL_TAG="py_gpu"
  else
    BAZEL_TAG="py_cpu"
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

function run_save_tensor_file {
  echo "Running in save tensor file mode: $@"
  XLA_SAVE_TENSORS_FILE="/tmp/xla_test_save_ir.txt" run_test "$@"
}

function run_xla_backend_mp {
  echo "Running XLA backend multiprocessing test: $@"
  MASTER_ADDR=localhost MASTER_PORT=6000 run_test "$@"
}

function run_xrt {
  if [ -x "$(command -v nvidia-smi)" ] && [ "$XLA_CUDA" != "0" ]; then
    GPU_NUM_DEVICES=2 run_coverage "$@"
  else
    XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0" XRT_WORKERS="localservice:0;grpc://localhost:$(shuf -i 40701-40999 -n 1)" run_coverage "$@"
  fi
}

function run_opbyop {
  echo "Running in OpByOp mode: $@"
  XLA_GET_TENSORS_OPBYOP=1 XLA_SYNC_TENSORS_OPBYOP=1 run_xrt "$@"
}

function run_async_scalar {
  echo "Running in Async Scalar Upload mode: $@"
  XLA_TRANSFER_SCALAR_ASYNC=1 run_xrt "$@"
}

function run_torchrun {
  echo "Running tests spawned by torchrun"
  if [ -x "$(command -v nvidia-smi)" ]; then
    run_xrt "$@"
  else
    echo "the tests need atleast two XLA workers to validate"
  fi
}

function run_xrt_tests {
  # For features not supported in PJRT
  echo "Running XRT tests"
  run_opbyop  "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_async_scalar  "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_torchrun  "$CDIR/test_allreduce_torchrun.py"
}

function run_torch_op_tests {
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
}

function set_bazel_caching() {
  EXTRA_FLAGS=""
  if [[ "$BAZEL_REMOTE_CACHE" == "1" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --config=remote_cache"
    if [[ ! -z "$GCLOUD_SERVICE_KEY_FILE" ]]; then
      EXTRA_FLAGS="$EXTRA_FLAGS --google_credentials=$GCLOUD_SERVICE_KEY_FILE"
    fi
    if [[ ! -z "$SILO_NAME" ]]; then
      EXTRA_FLAGS="$EXTRA_FLAGS --remote_default_exec_properties=cache-silo-key=$SILO_NAME"
    fi
  fi
  if [[ "$XLA_CUDA" == "1" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --config=cuda"
  fi
  if [[ "$BAZEL_VERB" == "coverage" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --remote_download_outputs=all" # for lcov symlink
  fi
}

function run_xla_op_tests {
  set_accelerator
  set_bazel_caching
  bazel $BAZEL_VERB $EXTRA_FLAGS --test_tag_filters="$BAZEL_TAG,-mp" --build_tag_filters=$BAZEL_TAG //test/... 
}

function run_op_tests {
  run_torch_op_tests
  run_xla_op_tests
}

function run_mp_op_tests {
  set_accelerator
  set_bazel_caching
  bazel $BAZEL_VERB $EXTRA_FLAGS --test_tag_filters="$BAZEL_TAG,-xla_op" --build_tag_filters=$BAZEL_TAG //test/... 
}

function run_tests {
  run_xla_op_tests
  if [[ "$XLA_SKIP_TORCH_OP_TESTS" != "1" ]]; then
    run_torch_op_tests
  fi
  if [[ "$XLA_SKIP_MP_OP_TESTS" != "1" ]]; then
    run_mp_op_tests
  fi
  if [[ "$XLA_SKIP_XRT_TESTS" != "1" ]]; then
    run_xrt_tests
  fi
}

if [ "$LOGFILE" != "" ]; then
  run_tests 2>&1 | tee $LOGFILE
else
  run_tests
fi
