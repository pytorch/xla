#!/bin/bash
set -xue
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
TEST_CDIR="$(dirname "$CDIR")"

source "${TEST_CDIR}/utils/run_tests_utils.sh"

# TODO: merge with other run_tests
python3 "$TEST_CDIR/test_operations.py" -v
python3 "$TEST_CDIR/pjrt/test_runtime_tpu.py"
python3 "$TEST_CDIR/pjrt/test_collective_ops_tpu.py"
python3 "$TEST_CDIR/spmd/test_mp_input_sharding.py"
run_save_tensor_hlo python3 "$TEST_CDIR/spmd/test_spmd_lowering_context.py"
python3 "$TEST_CDIR/spmd/test_xla_sharding.py"
python3 "$TEST_CDIR/spmd/test_xla_virtual_device.py"
python3 "$TEST_CDIR/spmd/test_xla_distributed_checkpoint.py"
python3 "$TEST_CDIR/spmd/test_train_spmd_linear_model.py"
python3 "$TEST_CDIR/spmd/test_xla_spmd_python_api_interaction.py"
python3 "$TEST_CDIR/spmd/test_xla_auto_sharding.py"
python3 "$TEST_CDIR/spmd/test_fsdp_v2.py"
XLA_EXPERIMENTAL=nonzero:masked_select:nms python3 "$TEST_CDIR/ds/test_dynamic_shape_models.py" -v
python3 "$TEST_CDIR/test_autocast.py"
python3 "$TEST_CDIR/test_fp8.py"
python3 "$TEST_CDIR/test_grad_checkpoint.py"
python3 "$TEST_CDIR/test_grad_checkpoint.py" "$@" --test_autocast
python3 "$TEST_CDIR/dynamo/test_dynamo.py"
python3 "$TEST_CDIR/dynamo/test_dynamo_dynamic_shape.py"
python3 "$TEST_CDIR/spmd/test_spmd_debugging.py"
XLA_PARAMETER_WRAPPING_THREADSHOLD=1 python3 "$TEST_CDIR/spmd/test_spmd_parameter_wrapping.py"
python3 "$TEST_CDIR/pjrt/test_dtypes.py"
python3 "$TEST_CDIR/pjrt/test_dynamic_plugin_tpu.py"
python3 "$TEST_CDIR/test_while_loop.py"
python3 "$TEST_CDIR/scan/test_scan.py"
python3 "$TEST_CDIR/scan/test_scan_spmd.py"
python3 "$TEST_CDIR/scan/test_scan_layers.py"
python3 "$TEST_CDIR/test_pallas.py" -v
python3 "$TEST_CDIR/test_pallas_spmd.py"
python3 "$TEST_CDIR/test_tpu_paged_attention_kernel.py"
python3 "$TEST_CDIR/test_input_output_aliases.py"
python3 "$TEST_CDIR/test_gmm.py"
python3 "$TEST_CDIR/eager/test_eager_spmd.py"
python3 "$TEST_CDIR/torch_distributed/test_torch_distributed_all_gather_xla_backend.py"
python3 "$TEST_CDIR/torch_distributed/test_torch_distributed_all_reduce_xla_backend.py"
python3 "$TEST_CDIR/torch_distributed/test_torch_distributed_multi_all_reduce_xla_backend.py"
python3 "$TEST_CDIR/torch_distributed/test_torch_distributed_reduce_scatter_xla_backend.py"
python3 "$TEST_CDIR/quantized_ops/test_dot_general.py"

# run examples, each test should takes <2 minutes
python3 "$TEST_CDIR/../examples/data_parallel/train_resnet_spmd_data_parallel.py"
python3 "$TEST_CDIR/../examples/fsdp/train_decoder_only_fsdp_v2.py"
python3 "$TEST_CDIR/../examples/train_resnet_amp.py"

# HACK: don't confuse local `torch_xla` folder with installed package
# Python 3.11 has the permanent fix: https://stackoverflow.com/a/73636559
# Egaer tests will take more HBM, only run them on TPU v4 CI
TPU_VERSION=$(python -c "import sys; sys.path.remove(''); import torch_xla; print(torch_xla._internal.tpu.version())")
if [[ -n "$TPU_VERSION" && "$TPU_VERSION" == "4" ]]; then
    python3 "$TEST_CDIR/dynamo/test_traceable_collectives.py"
    python3 "$TEST_CDIR/../examples/data_parallel/train_resnet_xla_ddp.py"
    python3 "$TEST_CDIR/../examples/fsdp/train_resnet_fsdp_auto_wrap.py"
    python3 "$TEST_CDIR/../examples/eager/train_decoder_only_eager.py"
    python3 "$TEST_CDIR/../examples/eager/train_decoder_only_eager_spmd_data_parallel.py"
    python3 "$TEST_CDIR/../examples/eager/train_decoder_only_eager_with_compile.py"
    python3 "$TEST_CDIR/../examples/eager/train_decoder_only_eager_multi_process.py"
    XLA_EXPERIMENTAL=nonzero:masked_select:nms python3 "$TEST_CDIR/ds/test_dynamic_shapes.py" -v
fi

if [[ -n "$TPU_VERSION" && "$TPU_VERSION" != "6" ]]; then
    # Test `tpu-info` CLI compatibility
    python3 "$CDIR/tpu_info/test_cli.py"
fi
