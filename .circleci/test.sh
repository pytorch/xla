#!/bin/bash

set -ex

source ./xla_env

if [ -x "$(command -v nvidia-smi)" ]; then
  export GPU_NUM_DEVICES=2
else
  export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  export XRT_WORKERS="localservice:0;grpc://localhost:40934"
fi

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export PYTORCH_TESTING_DEVICE_ONLY_FOR="xla"

cd /tmp/pytorch/xla

echo "Running Python Tests"
./test/run_tests.sh

# echo "Running MNIST Test"
# python test/test_train_mp_mnist.py --tidy
# if [ -x "$(command -v nvidia-smi)" ]; then
#   python test/test_train_mp_mnist_amp.py --fake_data
# fi

echo "Running C++ Tests"
pushd test/cpp
./run_tests.sh
if ! [ -x "$(command -v nvidia-smi)"  ]
then
  ./run_tests.sh -X early_sync -F AtenXlaTensorTest.TestEarlySyncLiveTensors -L""
fi
popd
