#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR=$CDIR/..
PTDIR=$XDIR/..

python $CDIR/gen.py \
  --gen_class_mode \
  --output_folder=$XDIR/torch_xla/csrc \
  $PTDIR/build/aten/src/ATen/TypeDefault.h \
  $PTDIR/build/aten/src/ATen/Functions.h \
  $PTDIR/build/aten/src/ATen/NativeFunctions.h
