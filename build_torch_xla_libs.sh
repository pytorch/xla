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

# VERBOSE=
# if [[ "$XLA_BAZEL_VERBOSE" == "1" ]]; then
#   VERBOSE="-s"
# fi
VERBOSE="-s"

TPUVM_FLAG=
if [[ "$TPUVM_MODE" == "1" ]]; then
  TPUVM_FLAG="--define=with_tpu_support=true"
fi

MAX_JOBS=
if [[ "$XLA_CUDA" == "1" ]] && [[ "$CLOUD_BUILD" == "true" ]]; then
  MAX_JOBS="--jobs=16"
fi

OPTS+=(--cxxopt="-std=c++14")
if [[ $(basename -- $CC) =~ ^clang ]]; then
  OPTS+=(--cxxopt="-Wno-c++11-narrowing")
fi

if [[ "$XLA_CUDA" == "1" ]]; then
  OPTS+=(--cxxopt="-DXLA_CUDA=1")
  OPTS+=(--config=cuda)
fi

if [ "$CMD" == "clean" ]; then
  pushd $THIRD_PARTY_DIR/tensorflow
  bazel clean
  popd
else
  cp -r -u -p $THIRD_PARTY_DIR/xla_client $THIRD_PARTY_DIR/tensorflow/tensorflow/compiler/xla/

  pushd $THIRD_PARTY_DIR/tensorflow
  bazel build $MAX_JOBS $VERBOSE $TPUVM_FLAG --define framework_shared_object=false -c "$MODE" "${OPTS[@]}" \
    $XLA_CUDA_CFG //tensorflow/compiler/xla/xla_client:libxla_computation_client.so

  popd
  mkdir -p torch_xla/lib
  chmod 0644 $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so
  cp $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so torch_xla/lib
fi
