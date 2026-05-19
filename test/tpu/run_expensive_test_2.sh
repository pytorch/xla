#!/bin/bash
set -xue
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
TEST_CDIR="$(dirname "$CDIR")"

# This test takes ~1000 seconds to run
python3 "$TEST_CDIR/test_ragged_paged_attention_kernel.py"
