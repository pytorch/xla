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

function pip_install() {
  # retry 3 times
  # old versions of pip don't have the "--progress-bar" flag
  pip install --progress-bar off "$@" || pip install --progress-bar off "$@" || pip install --progress-bar off "$@" ||\
  pip install "$@" || pip install "$@" || pip install "$@"
}

function install_torchvision() {
  # Check out torch/vision at Jun 11 2020 commit
  # This hash must match one in .jenkins/caffe2/test.sh
  pip_install --user git+https://github.com/pytorch/vision.git@c2e8a00885e68ae1200eb6440f540e181d9125de
}
