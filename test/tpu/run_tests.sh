#!/bin/bash
set -xue

# TODO: merge with other run_tests
python3 test/test_operations.py -v
python3 test/pjrt/test_runtime_tpu.py
python3 test/pjrt/test_collective_ops_tpu.py
python3 test/spmd/test_mp_input_sharding.py
python3 test/spmd/test_xla_sharding.py
python3 test/spmd/test_xla_virtual_device.py
python3 test/spmd/test_xla_distributed_checkpoint.py
python3 test/spmd/test_train_spmd_linear_model.py
python3 test/spmd/test_xla_spmd_python_api_interaction.py
python3 test/spmd/test_xla_auto_sharding.py
python3 test/spmd/test_fsdp_v2.py
XLA_EXPERIMENTAL=nonzero:masked_select:nms python3 test/ds/test_dynamic_shape_models.py -v
python3 test/test_autocast.py
python3 test/test_fp8.py
python3 test/test_grad_checkpoint.py
python3 test/dynamo/test_dynamo.py
python3 test/dynamo/test_dynamo_dynamic_shape.py
python3 test/spmd/test_spmd_debugging.py
XLA_PARAMETER_WRAPPING_THREADSHOLD=1 python test/spmd/test_spmd_parameter_wrapping.py
python3 test/pjrt/test_dtypes.py
python3 test/pjrt/test_dynamic_plugin_tpu.py
python3 test/test_while_loop.py
python3 test/test_scan.py
python3 test/test_pallas.py -v
python3 test/test_pallas_spmd.py
python3 test/test_tpu_paged_attention_kernel.py
python3 test/test_input_output_aliases.py
python3 test/test_gmm.py
python3 test/eager/test_eager_spmd.py
python3 test/torch_distributed/test_torch_distributed_all_gather_xla_backend.py
python3 test/torch_distributed/test_torch_distributed_all_reduce_xla_backend.py
python3 test/torch_distributed/test_torch_distributed_multi_all_reduce_xla_backend.py
python3 test/torch_distributed/test_torch_distributed_reduce_scatter_xla_backend.py
python3 test/quantized_ops/test_dot_general.py

# run examples, each test should takes <2 minutes
python3 examples/data_parallel/train_resnet_spmd_data_parallel.py
python3 examples/fsdp/train_decoder_only_fsdp_v2.py
python3 examples/train_resnet_amp.py

# HACK: don't confuse local `torch_xla` folder with installed package
# Python 3.11 has the permanent fix: https://stackoverflow.com/a/73636559
# Egaer tests will take more HBM, only run them on TPU v4 CI
TPU_VERSION=$(python -c "import sys; sys.path.remove(''); import torch_xla; print(torch_xla._internal.tpu.version())")
if [[ -n "$TPU_VERSION" && "$TPU_VERSION" == "4" ]]; then
    python3 test/dynamo/test_traceable_collectives.py
    python3 examples/data_parallel/train_resnet_xla_ddp.py
    python3 examples/fsdp/train_resnet_fsdp_auto_wrap.py
    python3 examples/eager/train_decoder_only_eager.py
    python3 examples/eager/train_decoder_only_eager_spmd_data_parallel.py
    python3 examples/eager/train_decoder_only_eager_with_compile.py
    python3 examples/eager/train_decoder_only_eager_multi_process.py
    XLA_EXPERIMENTAL=nonzero:masked_select:nms python3 test/ds/test_dynamic_shapes.py -v
fi

if [[ -n "$TPU_VERSION" && "$TPU_VERSION" != "6" ]]; then
    # Test `tpu-info` CLI compatibility
    python3 test/tpu/tpu_info/test_cli.py
fi
