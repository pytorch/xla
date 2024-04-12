#!/bin/bash

set -ex

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR=$CDIR/..
PTDIR=$XDIR/..
OPENXLADIR=$XDIR/third_party/xla

TORCH_PIN="$XDIR/.github/.torch_pin"
if [ -f "$TORCH_PIN" ]; then
  CID=$(cat "$TORCH_PIN")
  # If starts with # and it's not merged into master, fetch from origin
  if [[ $CID = \#* ]]; then
    PRNUM="${CID//[!0-9]/}"
    set +x
    MCHECK=$(git -C $PTDIR log -100)
    if [[ $MCHECK != *"Pull Request resolved: https://github.com/pytorch/pytorch/pull/$PRNUM"* ]]; then
      echo "Fetching PyTorch PR #$PRNUM"
      pushd "$PTDIR"
      git fetch origin "pull/$PRNUM/head:$PRNUM"
      git checkout "$PRNUM"
      git submodule update --init --recursive
      popd
    fi
    set -x
  elif [[ "$CID" != "" ]]; then
    echo 'Checking out branch $CID'
    pushd "$PTDIR"
    git fetch origin "$CID"
    git checkout "$CID"
    git submodule update --init --recursive
    popd
  fi
fi

python $CDIR/cond_patch.py \
  $XDIR/torch_patches \
  $PTDIR

# Apply OpenXLA patches only if requested, since bazel handles that normally.
if [[ -n "${APPLY_OPENXLA_patches}" ]]; then
  python $CDIR/cond_patch.py \
    $XDIR/openxla_patches \
    $OPENXLADIR
fi
