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

pip install --user https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl \
  https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-nightly-cp38-cp38-linux_x86_64.whl \
  https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl

set +e
build_torch_xla $XLA_DIR
pushd $XLA_DIR
LOGFILE=/tmp/pytorch_py_test.log
for FILE in $(ls tests/stablehlo); do
    python tests/stablehlo/$FILE | tee $LOGFILE
done



