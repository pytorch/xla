#!/bin/bash
set -xue
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
TEST_CDIR="$(dirname "$CDIR")"

# This test takes ~370 seconds to run
python3 "$TEST_CDIR/test_pallas.py" -v

# This test takes ~15 seconds to run
python3 "$TEST_CDIR/test_pallas_spmd.py"

# This test takes ~15 seconds to run
XLA_DISABLE_FUNCTIONALIZATION=1 python3 "$TEST_CDIR/test_pallas_spmd.py"
