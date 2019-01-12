#!/bin/bash

set -ex

cd /tmp/pytorch/xla

export XLA_USE_XRT=1 XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
export XRT_WORKERS="localservice:0;grpc://localhost:40934"

python test/test_operations.py
python test/test_train_mnist.py

