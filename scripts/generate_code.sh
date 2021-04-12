#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR="$CDIR/.."
PTDIR="$XDIR/.."
if [ -z "$PT_INC_DIR" ]; then
  PT_INC_DIR="$PTDIR/build/aten/src/ATen"
fi

python "$PTDIR/tools/codegen/gen_backend_stubs.py" \
  --output_dir="$XDIR/torch_xla/csrc" \
  --source_yaml="xla_native_functions.yaml"\
