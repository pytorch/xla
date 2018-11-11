#!/usr/bin/env bash

set -ex

cd "$(dirname "$0")"
PWD=`printf "%q\n" "$(pwd)"`
BASE_DIR="$PWD"
echo $BASE_DIR
THIRD_PARTY_DIR="$BASE_DIR/third_party"

cp -r -f $THIRD_PARTY_DIR/xla_client $THIRD_PARTY_DIR/tensorflow/tensorflow/compiler/xla/

pushd $THIRD_PARTY_DIR/tensorflow
git reset --hard
git clean -f
bazel build -c opt //tensorflow/compiler/xla/xla_client:libxla_computation_client.so
popd

mkdir -p torch_xla/lib
chmod 0644 $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so
cp $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/compiler/xla/xla_client/libxla_computation_client.so torch_xla/lib
chmod 0644 $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so
cp $THIRD_PARTY_DIR/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so torch_xla/lib
