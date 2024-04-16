#!/bin/bash

set -ex

source .circleci/common.sh
PYTORCH_DIR=/tmp/pytorch
XLA_DIR=$PYTORCH_DIR/xla
clone_pytorch $PYTORCH_DIR $XLA_DIR

# Use bazel cache
USE_CACHE=1

pushd $PYTORCH_DIR
checkout_torch_pin_if_available

if ! install_deps_pytorch_xla $XLA_DIR $USE_CACHE; then
  exit 1
fi

apply_patches

python -c "import fcntl; fcntl.fcntl(1, fcntl.F_SETFL, 0)"

export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64
export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST='8.6'
python setup.py install

XLA_DIR=$PYTORCH_DIR/xla
export TF_CUDA_COMPUTE_CAPABILITIES="compute_86"
export XLA_CUDA=1
build_torch_xla $XLA_DIR