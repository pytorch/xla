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
