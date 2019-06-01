#!/bin/bash
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
export XRT_WORKERS="localservice:0;grpc://localhost:40934"

time python pytorch/xla/test/test_train_mnist.py
time bash pytorch/xla/test/run_tests.sh
time bash pytorch/xla/test/cpp/run_tests.sh
