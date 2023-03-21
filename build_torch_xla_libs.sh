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

if [[ "$TPUVM_MODE" == "1" ]]; then
  OPTS+=(--config=tpu)
fi

MAX_JOBS=
if [[ ! -z "$BAZEL_JOBS" ]]; then
  MAX_JOBS="--jobs=$BAZEL_JOBS"
fi

if [[ "$XLA_CUDA" == "1" ]]; then
  OPTS+=(--config=cuda)
fi

if [[ "$XLA_CPU_USE_ACL" == "1" ]]; then
  OPTS+=(--config=acl)
fi

if [ "$CMD" == "clean" ]; then
  bazel clean
  exit 0
fi

# TensorFlow and its dependencies may introduce warning flags from newer compilers
# that PyTorch and PyTorch/XLA's default compilers don't recognize. They become error
# while '-Werror' is used. Therefore, surpress the warnings in .bazelrc or here.
bazel build $MAX_JOBS $VERBOSE --show_progress_rate_limit=20 \
  --define framework_shared_object=false -c "$MODE" "${OPTS[@]}" \
  $XLA_CUDA_CFG //third_party/xla_client:xla_computation_client

mkdir -p torch_xla/lib
chmod 0644 bazel-bin/third_party/xla_client/libxla_computation_client.so
cp bazel-bin/third_party/xla_client/libxla_computation_client.so torch_xla/lib
