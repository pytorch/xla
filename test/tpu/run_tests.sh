#!/bin/bash
set -xue

# TODO: merge with other run_tests
python3 test/test_operations.py -v
python3 test/pjrt/test_runtime_tpu.py
python3 test/pjrt/test_collective_ops_tpu.py
python3 test/spmd/test_xla_sharding.py
python3 test/spmd/test_xla_virtual_device.py
python3 test/spmd/test_xla_distributed_checkpoint.py
python3 test/spmd/test_train_spmd_linear_model.py
python3 test/spmd/test_xla_spmd_python_api_interaction.py
python3 test/spmd/test_xla_auto_sharding.py
XLA_EXPERIMENTAL=nonzero:masked_select python3 test/ds/test_dynamic_shape_models.py -v
XLA_EXPERIMENTAL=nonzero:masked_select python3 test/ds/test_dynamic_shapes.py -v
python3 test/test_autocast.py
python3 test/dynamo/test_dynamo.py
python3 test/spmd/test_spmd_debugging.py
python3 test/pjrt/test_dtypes.py
python3 test/pjrt/test_dynamic_plugin_tpu.py
python3 test/test_fori_loop_with_while_loop_simple_add_dispatch_in_torch.py
python3 test/test_pallas.py
python3 test/torch_distributed/test_torch_distributed_all_gather_xla_backend.py
python3 test/torch_distributed/test_torch_distributed_all_reduce_xla_backend.py
python3 test/torch_distributed/test_torch_distributed_multi_all_reduce_xla_backend.py
python3 test/torch_distributed/test_torch_distributed_reduce_scatter_xla_backend.py
