#!/bin/bash

set -e  # Fail on any error.
set -x  # Display commands being run.

# Disable prompts.
export DEBIAN_FRONTEND=noninteractive

function install_bazel() {
  local BAZEL_VERSION="0.24.1"
  local BAZEL_FILE="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
  sudo apt-get install pkg-config zip zlib1g-dev unzip
  curl -L -O "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${BAZEL_FILE}"
  chmod 755 "$BAZEL_FILE"
  ./"$BAZEL_FILE" --user
  export PATH="$PATH:$HOME/bin"
}

function install_clang() {
  sudo bash -c 'echo "deb http://deb.debian.org/debian/ testing main" >> /etc/apt/sources.list'
  sudo apt-get update
  sudo apt-get install -y clang-7 clang++-7
  export CC=clang-7 CXX=clang++-7
}

function install_req_packages() {
  sudo apt-get -y install python-pip git libopenblas-dev
  sudo pip install --upgrade google-api-python-client
  sudo pip install --upgrade oauth2client
  install_bazel
}

function install_and_setup_conda() {
  curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
  sh Anaconda3-5.2.0-Linux-x86_64.sh -b
  export PATH="$HOME/anaconda3/bin:$PATH"
  conda create --name pytorch python=3.5 anaconda
  source activate pytorch
  export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
  conda install -y numpy pyyaml setuptools cmake cffi typing
  sudo /sbin/ldconfig "${HOME}/anaconda3/lib/" "${HOME}/anaconda3/envs/pytorch/lib"
  pip install lark-parser
}

function build_and_install_torch() {
  git submodule update --init --recursive
  # Checkout the PT commit ID if we have one.
  COMMITID_FILE="xla/.torch_commit_id"
  if [ -e "$COMMITID_FILE" ]; then
    git checkout $(cat "$COMMITID_FILE")
  fi
  # Apply patches to PT which are required by the XLA support.
  xla/scripts/apply_patches.sh
  # Build and install torch wheel and collect artifact
  export NO_CUDA=1 NO_MKLDNN=1
  python setup.py bdist_wheel
  pip install dist/*.whl
}

function build_and_install_torch_xla() {
  git submodule update --init --recursive
  export NO_CUDA=1 VERSIONED_XLA_BUILD=1
  python setup.py bdist_wheel
  pip install dist/*.whl
}

function main() {
  install_clang
  install_req_packages
  install_and_setup_conda
  build_and_install_torch
  cd xla
  build_and_install_torch_xla
}

main
