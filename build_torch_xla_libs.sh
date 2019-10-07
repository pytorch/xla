#!/usr/bin/env bash

set -ex

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

XLA_CUDA_CFG=
if [[ "$XLA_CUDA" == "1" ]]; then
  XLA_CUDA_CFG="--config=cuda"
fi

OPTS=(--cxxopt="-std=c++14")
if [[ "$CC" =~ ^clang ]]; then
  OPTS+=(--cxxopt="-Wno-c++11-narrowing")
fi

if [ "$CMD" == "clean" ]; then
  pushd $THIRD_PARTY_DIR/tensorflow
  bazel clean
  popd
else
  cp -r -u -p $THIRD_PARTY_DIR/xla_client $THIRD_PARTY_DIR/tensorflow/tensorflow/compiler/xla/

  pushd $THIRD_PARTY_DIR/tensorflow
  bazel build --define framework_shared_object=false -c "$MODE" "${OPTS[@]}" $XLA_CUDA_CFG \
    //tensorflow/compiler/xla/xla_client:libxla_computation_client.so

  popd
  mkdir -p torch_xla/lib
  chmod 0644 $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so
  cp $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so torch_xla/lib
fi
