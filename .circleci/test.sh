#!/bin/bash

set -ex

cd /tmp/pytorch/xla

source ./xla_env

python test/test_operations.py
python test/test_train_mnist.py --tidy

pushd test/cpp
./run_tests.sh
popd
