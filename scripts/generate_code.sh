#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR="$CDIR/.."
PTDIR="$XDIR/.."
if [ -z "$PT_INC_DIR" ]; then
  PT_INC_DIR="$PTDIR/build/aten/src/ATen"
fi

python "$CDIR/gen.py" \
  --gen_class_mode \
  --output_folder="$XDIR/torch_xla/csrc" \
  "$PT_INC_DIR/TypeDefault.h" \
  "$PT_INC_DIR/Functions.h" \
  "$PT_INC_DIR/NativeFunctions.h"
