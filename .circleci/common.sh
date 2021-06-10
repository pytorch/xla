#!/bin/bash

set -ex

# System default cmake 3.10 cannot find mkl, so point it to the right place.
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

function clone_pytorch() {
  PYTORCH_DIR=$1
  XLA_DIR=$2
  git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
  cp -r "$PWD" "$XLA_DIR"
}

function run_torch_xla_tests() {
  PYTORCH_DIR=$1
  XLA_DIR=$2
  if [ -x "$(command -v nvidia-smi)" ]; then
    export GPU_NUM_DEVICES=2
  else
    export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
    XLA_PORT=$(shuf -i 40701-40999 -n 1)
    export XRT_WORKERS="localservice:0;grpc://localhost:$XLA_PORT"
  fi
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="xla"

  pushd $XLA_DIR
    echo "Running Python Tests"
    ./test/run_tests.sh

    # echo "Running MNIST Test"
    # python test/test_train_mp_mnist.py --tidy
    # if [ -x "$(command -v nvidia-smi)" ]; then
    #   python test/test_train_mp_mnist_amp.py --fake_data
    # fi

    pushd test/cpp
    CC=clang-9 CXX=clang++-9 ./run_tests.sh

    if ! [ -x "$(command -v nvidia-smi)"  ]
    then
      ./run_tests.sh -X early_sync -F AtenXlaTensorTest.TestEarlySyncLiveTensors -L""
    fi
    popd
  popd
}
