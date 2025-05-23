#!/bin/bash
set -xue

# Absolute path to the directory of this script.
_TPU_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"

# Absolute path to the test/ directory.
_TEST_DIR="$(dirname "$_TPU_DIR")"

# This test takes ~370 seconds to run
python3 "$_TEST_DIR/test_pallas.py" -v

# This test takes ~15 seconds to run
python3 "$_TEST_DIR/test_pallas_spmd.py"

# This test takes ~15 seconds to run
XLA_DISABLE_FUNCTIONALIZATION=1 python3 "$_TEST_DIR/test_pallas_spmd.py"
