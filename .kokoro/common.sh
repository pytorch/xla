#!/bin/bash

set -ex

DEBIAN_FRONTEND=noninteractive

function apply_patches() {
  # assumes inside pytorch dir
  ./xla/scripts/apply_patches.sh
}

function checkout_torch_pin_if_available() {
  COMMITID_FILE="xla/.torch_pin"
  if [ -e "$COMMITID_FILE" ]; then
    git checkout $(cat "$COMMITID_FILE")
  fi
  git submodule update --init --recursive
}

function install_deps_pytorch_xla() {
  XLA_DIR=$1

  # Install pytorch deps
  pip install sympy

  # Install ninja to speedup the build
  pip install ninja

  # Install libraries required for running some PyTorch test suites
  pip install hypothesis
  pip install cloud-tpu-client
  pip install absl-py
  pip install --upgrade "numpy>=1.18.5"
  pip install --upgrade numba

  # Using the Ninja generator requires CMake version 3.13 or greater
  pip install "cmake>=3.13" --upgrade

  sudo apt-get -qq update
}

function build_torch_xla() {
  XLA_DIR=$1
  pushd "$XLA_DIR"
  XLA_CUDA=0 python setup.py install
  popd
}

function pip_install() {
  # retry 3 times
  # old versions of pip don't have the "--progress-bar" flag
  pip install --progress-bar off "$@" || pip install --progress-bar off "$@" || pip install --progress-bar off "$@" ||\
  pip install "$@" || pip install "$@" || pip install "$@"
}

function install_torchvision() {
  pip_install --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@$TORCHVISION_COMMIT"
}

function run_torch_xla_tests() {
  PYTORCH_DIR=$1
  XLA_DIR=$2

  pushd $XLA_DIR
    echo "Running integration tests..."
    XLA_SKIP_TORCH_OP_TESTS=1 ./test/run_tests.sh
  popd
}
