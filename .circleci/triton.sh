#!/bin/bash

set -ex

source ./xla_env
source .circleci/common.sh

PYTORCH_DIR=/tmp/pytorch
pushd $PYTORCH_DIR
export TORCH_CUDA_ARCH_LIST="8.6"
python setup.py install

XLA_DIR=$PYTORCH_DIR/xla
export TF_CUDA_COMPUTE_CAPABILITIES="compute_86"
build_torch_xla $XLA_DIR

export GCLOUD_SERVICE_KEY_FILE="$XLA_DIR/default_credentials.json"
export SILO_NAME='cache-silo-ci-dev-3.8_cuda_12.1'  # cache bucket for CI
export TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas'
python3 $XLA_DIR/test/test_triton.py