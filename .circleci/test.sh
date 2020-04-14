#!/bin/bash

set -ex

source ./xla_env

conda activate base
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

cd /tmp/pytorch/xla

echo "Running Python Tests"
./test/run_tests.sh

echo "Running MNIST Test"
python test/test_train_mnist.py --tidy

echo "Running C++ Tests"
pushd test/cpp
./run_tests.sh
popd
