#!/bin/bash

set -ex

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

  sudo apt-get -qq install npm nodejs

  # XLA build requires Bazel
  # We use bazelisk to avoid updating Bazel version manually.
  sudo npm install -g @bazel/bazelisk
  # Only unlink if file exists
  if [[ -e /usr/bin/bazel ]]; then
    sudo unlink /usr/bin/bazel
  fi

  sudo ln -s "$(command -v bazelisk)" /usr/bin/bazel

  # Symnlink the missing cuda headers if exists
  CUBLAS_PATTERN="/usr/include/cublas*"
  if ls $CUBLAS_PATTERN 1> /dev/null 2>&1; then
    sudo ln -s $CUBLAS_PATTERN /usr/local/cuda/include
  fi
}

function build_torch_xla() {
  XLA_DIR=$1
  pushd "$XLA_DIR"
  python setup.py install
  popd
}

function run_torch_xla_tests() {
  PYTORCH_DIR=$1
  XLA_DIR=$2

  pushd $XLA_DIR
    echo "Running integration tests..."
    XLA_SKIP_TORCH_OP_TESTS=1 ./test/run_tests.sh
  popd
}
