#!/bin/bash
set -xue

# Absolute path to the directory of this script.
_TPU_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"

# Absolute path to the test/ directory.
_TEST_DIR="$(dirname "$_TPU_DIR")"

# run examples, each test should takes <2 minutes
python3 "$_TEST_DIR/../examples/data_parallel/train_resnet_spmd_data_parallel.py"
python3 "$_TEST_DIR/../examples/fsdp/train_decoder_only_fsdp_v2.py"
python3 "$_TEST_DIR/../examples/train_resnet_amp.py"
python3 "$_TEST_DIR/../examples/train_decoder_only_base.py"
python3 "$_TEST_DIR/../examples/train_decoder_only_base.py" scan.decoder_with_scan.DecoderWithScan \
    --num-steps 30 # TODO(https://github.com/pytorch/xla/issues/8632): Reduce scan tracing overhead

# HACK: don't confuse local `torch_xla` folder with installed package
# Python 3.11 has the permanent fix: https://stackoverflow.com/a/73636559
# Egaer tests will take more HBM, only run them on TPU v4 CI
TPU_VERSION=$(python -c "import sys; sys.path.remove(''); import torch_xla; print(torch_xla._internal.tpu.version())")
if [[ -n "$TPU_VERSION" && "$TPU_VERSION" == "4" ]]; then
    python3 "$_TEST_DIR/dynamo/test_traceable_collectives.py"
    python3 "$_TEST_DIR/../examples/data_parallel/train_resnet_xla_ddp.py"
    python3 "$_TEST_DIR/../examples/fsdp/train_resnet_fsdp_auto_wrap.py"
    python3 "$_TEST_DIR/../examples/eager/train_decoder_only_eager.py"
    python3 "$_TEST_DIR/../examples/eager/train_decoder_only_eager_spmd_data_parallel.py"
    python3 "$_TEST_DIR/../examples/eager/train_decoder_only_eager_with_compile.py"
    python3 "$_TEST_DIR/../examples/eager/train_decoder_only_eager_multi_process.py"
    XLA_EXPERIMENTAL=nonzero:masked_select:nms python3 "$_TEST_DIR/ds/test_dynamic_shapes.py" -v
fi

if [[ -n "$TPU_VERSION" && "$TPU_VERSION" != "6" ]]; then
    # Test `tpu-info` CLI compatibility
    python3 "$_TPU_DIR/tpu_info/test_cli.py"
fi
