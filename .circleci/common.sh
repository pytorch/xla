#!/bin/bash

set -ex

function setup_env() {
  export PYTORCH_DIR=/tmp/pytorch
  export XLA_DIR="$PYTORCH_DIR/xla"
}

function clone_pytorch() {
  setup_env
  git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
  cp -r "$PWD" "$XLA_DIR"
}
