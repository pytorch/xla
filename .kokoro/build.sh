#!/bin/bash

set -e

PYTORCH_DIR="${KOKORO_ARTIFACTS_DIR}/github/pytorch"
XLA_DIR=$PYTORCH_DIR/xla
git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
cp -r "${KOKORO_ARTIFACTS_DIR}/github/xla" "$XLA_DIR"
source ${XLA_DIR}/.kokoro/common.sh

pushd $PYTORCH_DIR
checkout_torch_pin_if_available
install_deps_pytorch_xla $XLA_DIR
apply_patches

python -c "import fcntl; fcntl.fcntl(1, fcntl.F_SETFL, 0)"
python setup.py install

source $XLA_DIR/xla_env
build_torch_xla $XLA_DIR

popd

