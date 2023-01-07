#!/bin/bash
set -ex

export TF_CPP_LOG_THREAD_ID=1
export WITH_XLA=1 NO_CUDA=1 XLA_DEBUG_GC=2 CC=clang-8 CXX=clang++-8 CLANG_FORMAT="clang-format-7"
export COMPILE_PARALLEL=1 DEBUG=1 XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 TF_CPP_LOG_THREAD_ID=1
export XRT_WORKERS="localservice:0;grpc://localhost:51011"
export XLA_USE_XRT=1 XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0" 

export XLA_CUDA=1
export GITHUB_TORCH_XLA_BOT_TOKEN=84eaa41dc4141a6b3fe81c3097347b2e148342c6

export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-${CONDA_PREFIX:-"$(dirname $(which conda))/../"}}

export PYTORCH_DIR=/pytorch
export XLA_DIR=$PYTORCH_DIR/xla

# Needs to be kept in sync with .jenkins/pytorch/common_utils.sh in pytorch/pytorch.
export TORCHVISION_COMMIT="$(cat $PYTORCH_DIR/.github/ci_commit_pins/vision.txt)"

# run_torch_xla_tests
XLA_PORT=$(shuf -i 40701-40999 -n 1)
export XRT_WORKERS="localservice:0;grpc://localhost:$XLA_PORT"

export PYTORCH_TESTING_DEVICE_ONLY_FOR="xla"

pushd $XLA_DIR
  echo "Running Python Tests"
  ./test/run_tests.sh
popd

echo "Finish running all python op tests"
