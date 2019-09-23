#!/bin/bash

set -ex

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR=$CDIR/..
PTDIR=$XDIR/..

python $CDIR/cond_patch.py \
  $XDIR/torch_patches \
  $PTDIR
