#!/usr/bin/env bash

set -ex

CMD="${1:-install}"

cd "$(dirname "$0")"
PWD=`printf "%q\n" "$(pwd)"`
BASE_DIR="$PWD"
echo $BASE_DIR
THIRD_PARTY_DIR="$BASE_DIR/third_party"

OPTS=(--cxxopt="-std=c++14")
if [[ "$CC" =~ ^clang ]]; then
  OPTS+=(--cxxopt="-Wno-c++11-narrowing")
fi

if [ "$CMD" == "clean" ]; then
  pushd $THIRD_PARTY_DIR/tensorflow
  bazel clean
  popd
else
  cp -r -f $THIRD_PARTY_DIR/xla_client $THIRD_PARTY_DIR/tensorflow/tensorflow/compiler/xla/

  pushd $THIRD_PARTY_DIR/tensorflow
  git reset --hard
  git clean -f
  bazel build --define framework_shared_object=false -c opt "${OPTS[@]}" \
    //tensorflow/compiler/xla/xla_client:libxla_computation_client.so

  popd
  mkdir -p torch_xla/lib
  chmod 0644 $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so
  cp $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so torch_xla/lib
fi
