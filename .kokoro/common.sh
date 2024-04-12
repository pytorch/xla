#!/bin/bash

set -ex

DEBIAN_FRONTEND=noninteractive

function apply_patches() {
  # assumes inside pytorch dir
  ./xla/scripts/apply_patches.sh
}

function checkout_torch_pin_if_available() {
  COMMITID_FILE="xla/.github/.torch_pin"
  if [ -e "$COMMITID_FILE" ]; then
    git checkout $(cat "$COMMITID_FILE")
  fi
  git submodule update --init --recursive
}

function install_environments() {
  apt-get clean && apt-get update
  apt-get upgrade -y
  apt-get install --fix-missing -y python3-pip git curl libopenblas-dev vim jq \
    apt-transport-https ca-certificates procps openssl sudo wget libssl-dev libc6-dbg

  curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
  mv bazelisk-linux-amd64 /usr/local/bin/bazel
  chmod +x /usr/local/bin/bazel

  pip install mkl mkl-include setuptools typing_extensions cmake requests
  sudo ln -sf /usr/local/lib/libmkl_intel_lp64.so.2 /usr/local/lib/libmkl_intel_lp64.so.1
  sudo ln -sf /usr/local/lib/libmkl_intel_thread.so.2 /usr/local/lib/libmkl_intel_thread.so.1
  sudo ln -sf /usr/local/lib/libmkl_core.so.2 /usr/local/lib/libmkl_core.so.1
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" >> /etc/apt/sources.list.d/google-cloud-sdk.list
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

  apt-get update
  apt-get -y install google-cloud-cli
  pip install --upgrade google-api-python-client
  pip install --upgrade oauth2client
  pip install --upgrade google-cloud-storage
  pip install lark-parser
  pip install cloud-tpu-client
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
  pip install pandas
  pip install tabulate
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
    # Run integration tests with CPU
    XLA_CUDA=0 USE_COVERAGE=0 XLA_SKIP_TORCH_OP_TESTS=1 XLA_SKIP_XRT_TESTS=1 CONTINUE_ON_ERROR=1 ./test/run_tests.sh
  popd
}
