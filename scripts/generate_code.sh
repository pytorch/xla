#!/bin/bash

CDIR=$(dirname $0)
XDIR=$(realpath $CDIR/..)
PTDIR=$(realpath $XDIR/..)

$CDIR/gen.py \
  --gen_class_mode \
  --output_folder=$XDIR/torch_xla/csrc \
  $PTDIR/build/aten/src/ATen/TypeDefault.h \
  $PTDIR/build/aten/src/ATen/Functions.h \
  $PTDIR/build/aten/src/ATen/NativeFunctions.h

