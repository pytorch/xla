#!/bin/bash

set -ex

export PYTORCH_DIR=/tmp/pytorch
export XLA_DIR="$PYTORCH_DIR/xla"

function clone_pytorch() {
  git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
  cp -r "$PWD" "$XLA_DIR"
}
