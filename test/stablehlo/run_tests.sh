#!/bin/bash
# the test script runs tests not covered by TPU CI
# SPMD
PJRT_DEVICE=TPU python3 test/spmd/test_xla_sharding.py
PJRT_DEVICE=TPU python3 test/spmd/test_xla_virtual_device.py
PJRT_DEVICE=TPU python3 test/spmd/test_train_spmd_linear_model.py
# CPP tests
PJRT_DEVICE=TPU XLA_EXPERIMENTAL="nonzero:masked_select" ./test/cpp/build/ptxla
# Dynamo
PJRT_DEVICE=TPU python3 test/dynamo/test_bridge.py
PJRT_DEVICE=TPU python3 test/dynamo/test_dynamo.py
PJRT_DEVICE=TPU python3 test/dynamo/test_num_output.py
# PJRT
PJRT_DEVICE=TPU python3 test/pjrt/test_ddp.py
PJRT_DEVICE=TPU python3 test/pjrt/test_experimental_pjrt_tpu.py
PJRT_DEVICE=TPU python3 test/pjrt/test_experimental_tpu.py
PJRT_DEVICE=TPU python3 test/pjrt/test_mesh_service.py
