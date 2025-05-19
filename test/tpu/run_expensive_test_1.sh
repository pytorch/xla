#!/bin/bash
set -xue
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
TEST_CDIR="$(dirname "$CDIR")"

# This test takes ~1350 seconds to run
python3 "$TEST_CDIR/test_multi_queries_paged_attention_kernel.py"
