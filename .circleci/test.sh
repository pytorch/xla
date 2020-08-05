#!/bin/bash

set -ex

source ./xla_env

if [ -x "$(command -v nvidia-smi)" ]; then
  export GPU_NUM_DEVICES=2
else
  export CPU_NUM_DEVICES=4
fi

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

cd /tmp/pytorch/xla

echo "Running Python Tests"
./test/run_tests.sh

echo "Running MNIST Test"
python test/test_train_mp_mnist.py --tidy

echo "Running C++ Tests"
pushd test/cpp
./run_tests.sh
popd
