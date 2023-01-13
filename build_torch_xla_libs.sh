#!/usr/bin/env bash

set -ex

OPTS=()

while getopts "O:" OPTION
do
    case $OPTION in
       O)
           for i in ${OPTARG}; do
               OPTS+=("--cxxopt=${i}")
           done
           ;;
    esac
done
shift $(($OPTIND - 1))

CMD="${1:-install}"

cd "$(dirname "$0")"
PWD=`printf "%q\n" "$(pwd)"`
BASE_DIR="$PWD"
echo $BASE_DIR
THIRD_PARTY_DIR="$BASE_DIR/third_party"

MODE="opt"
if [[ "$XLA_DEBUG" == "1" ]]; then
  MODE="dbg"
fi

VERBOSE=
if [[ "$XLA_BAZEL_VERBOSE" == "1" ]]; then
  VERBOSE="-s"
fi

BUILD_STRATEGY="standalone"
if [[ "$XLA_SANDBOX_BUILD" == "1" ]]; then
  BUILD_STRATEGY="sandboxed --sandbox_base=/dev/shm"
else
  # We can remove this after https://github.com/bazelbuild/bazel/issues/15359 is resolved
  unset CC
  unset CXX
  BUILD_STRATEGY="local"
fi

TPUVM_FLAG=
if [[ "$TPUVM_MODE" == "1" ]]; then
  TPUVM_FLAG="--define=with_tpu_support=true"
fi

MAX_JOBS=
if [[ ! -z "$BAZEL_JOBS" ]]; then
  MAX_JOBS="--jobs=$BAZEL_JOBS"
fi

OPTS+=(--cxxopt="-std=c++17")
if [[ $(basename -- $CC) =~ ^clang ]]; then
  OPTS+=(--cxxopt="-Wno-c++11-narrowing")
  OPTS+=(--cxxopt="-Wno-c++14-narrowing")
fi

if [[ "$XLA_CUDA" == "1" ]]; then
  OPTS+=(--cxxopt="-DXLA_CUDA=1")
  OPTS+=(--config=cuda)
fi

if [[ "$XLA_CPU_USE_ACL" == "1" ]]; then
  OPTS+=("--define=build_with_acl=true")
fi

if [ "$CMD" == "clean" ]; then
  pushd $THIRD_PARTY_DIR/tensorflow
  bazel clean
  popd
else
  # Overlay llvm-raw secondary cache. The remote cache should be updated
  # nightly with the pinned llvm archive. Note, this commands will be NO-OP if there is no match.
  sed -i '/.*github.com\/llvm.*,/a "https://storage.googleapis.com/tpu-pytorch/llvm-raw/{commit}.tar.gz".format(commit = LLVM_COMMIT),' \
    $THIRD_PARTY_DIR/tensorflow/third_party/llvm/workspace.bzl
  sed -i 's/LLVM_COMMIT)]/LLVM_COMMIT),"https:\/\/storage.googleapis.com\/tpu-pytorch\/llvm-raw\/{commit}.tar.gz".format(commit = LLVM_COMMIT)]/g' \
    $THIRD_PARTY_DIR/tensorflow/tensorflow/compiler/xla/mlir_hlo/WORKSPACE

  cp -r -u -p $THIRD_PARTY_DIR/xla_client $THIRD_PARTY_DIR/tensorflow/tensorflow/compiler/xla/

  pushd $THIRD_PARTY_DIR/tensorflow
  # TensorFlow and its dependencies may introduce warning flags from newer compilers 
  # that PyTorch and PyTorch/XLA's default compilers don't recognize. They become error
  # while '-Werror' is used. Therefore, surpress the warnings.
  TF_EXTRA_FLAGS="--copt=-Wno-unknown-warning-option"
  bazel build $MAX_JOBS $VERBOSE $TPUVM_FLAG $TF_EXTRA_FLAGS --spawn_strategy=$BUILD_STRATEGY --show_progress_rate_limit=20 \
    --define framework_shared_object=false -c "$MODE" "${OPTS[@]}" \
    $XLA_CUDA_CFG //tensorflow/compiler/xla/xla_client:libxla_computation_client.so

  popd
  mkdir -p torch_xla/lib
  chmod 0644 $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so
  cp $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so torch_xla/lib
fi
