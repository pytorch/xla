#!/usr/bin/env bash

set -euo pipefail

function setup_env() {
  export PYTORCH_DIR='/tmp/pytorch'
  export XLA_DIR="$PYTORCH_DIR"'/xla'
}

function clone_pytorch() {
  setup_env
    
  if [ -d "$PYTORCH_DIR"'/.git' ]; then
    GIT_DIR="$PYTORCH_DIR"'/.git' GIT_WORK_TREE="$PYTORCH_DIR" git pull
  else
    git clone --quiet --depth=10 https://github.com/pytorch/pytorch "$PYTORCH_DIR"
    cp -r "$PWD" "$XLA_DIR"
  fi
}
