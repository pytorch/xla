#!/bin/bash

set -ex

DEBIAN_FRONTEND=noninteractive

PYTORCH_DIR="${KOKORO_ARTIFACTS_DIR}/github/pytorch"
XLA_DIR=$PYTORCH_DIR/xla
git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
cp -r "${KOKORO_ARTIFACTS_DIR}/github/xla" "$XLA_DIR"
source ${XLA_DIR}/.kokoro/common.sh

TORCHVISION_COMMIT="$(cat $PYTORCH_DIR/.github/ci_commit_pins/vision.txt)"

install_environments

set +e
# build pytorch
pushd $PYTORCH_DIR
pip install --user .
popd
build_torch_xla $XLA_DIR
pushd $XLA_DIR
LOGFILE=/tmp/pytorch_py_test.log
TEST_DIR=test/stablehlo
for FILE in $(ls $TEST_DIR); do
    XLA_STABLEHLO_COMPILE=1 python $TEST_DIR/$FILE | tee $LOGFILE
done



