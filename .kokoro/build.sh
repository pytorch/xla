#!/bin/bash

set -e

DEBIAN_FRONTEND=noninteractive

PYTORCH_DIR="${KOKORO_ARTIFACTS_DIR}/github/pytorch"
XLA_DIR=$PYTORCH_DIR/xla
git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
cp -r "${KOKORO_ARTIFACTS_DIR}/github/xla" "$XLA_DIR"
source ${XLA_DIR}/.kokoro/common.sh

TORCHVISION_COMMIT="$(cat $PYTORCH_DIR/.github/ci_commit_pins/vision.txt)"

apt-get clean && apt-get update
apt-get upgrade -y
apt-get install --fix-missing -y python3-pip git curl libopenblas-dev vim jq \
  apt-transport-https ca-certificates procps openssl sudo wget libssl-dev libc6-dbg

curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
mv bazelisk-linux-amd64 /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

pushd $PYTORCH_DIR
checkout_torch_pin_if_available
install_deps_pytorch_xla $XLA_DIR
apply_patches

python -c "import fcntl; fcntl.fcntl(1, fcntl.F_SETFL, 0)"
USE_CUDA=0 python setup.py install

build_torch_xla $XLA_DIR

popd

install_torchvision
run_torch_xla_tests $PYTORCH_DIR $XLA_DIR


