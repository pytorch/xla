#!/bin/bash

set -ex

cd /tmp/pytorch/xla

source ./xla_env

echo "Running Python Tests"
./test/run_tests.sh

echo "Running MNIST Test"
python test/test_train_mnist.py --tidy

echo "Running C++ Tests"
pushd test/cpp
./run_tests.sh
popd
