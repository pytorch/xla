#!/bin/bash

CDIR=$(dirname $0)
XDIR=$CDIR/..
PTDIR=$XDIR/..

python $CDIR/cond_patch.py \
  $XDIR/torch_patches \
  $PTDIR
