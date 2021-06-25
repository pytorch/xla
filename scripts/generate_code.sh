#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR="$CDIR/.."
PTDIR="$XDIR/.."
if [ -z "$PT_INC_DIR" ]; then
  PT_INC_DIR="$PTDIR/build/aten/src/ATen"
fi

set -e
pushd $PTDIR
python -m tools.codegen.gen_backend_stubs \
  --output_dir="$XDIR/torch_xla/csrc" \
  --source_yaml="$XDIR/xla_native_functions.yaml"\
  --impl_path="$XDIR/torch_xla/csrc/aten_xla_type.cpp"\

popd
