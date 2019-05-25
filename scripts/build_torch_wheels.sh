#!/bin/bash

set -e  # Fail on any error.
set -x  # Display commands being run.

export DEBIAN_FRONTEND=noninteractive

function install_bazel() {
  local BAZEL_VERSION="0.24.1"
  local BAZEL_FILE="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
  sudo apt-get install -y pkg-config zip zlib1g-dev unzip
  curl -L -O "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${BAZEL_FILE}"
  chmod 755 "$BAZEL_FILE"
  ./"$BAZEL_FILE" --user
  export PATH="$PATH:$HOME/bin"
}

function install_clang() {
  if ! grep -Fxq "deb http://deb.debian.org/debian/ testing main" /etc/apt/sources.list; then
    sudo bash -c 'echo "deb http://deb.debian.org/debian/ testing main" >> /etc/apt/sources.list'
  fi
  sudo apt-get update
  sudo apt-get install -y clang-7 clang++-7
  export CC=clang-7 CXX=clang++-7
}

function install_req_packages() {
  sudo apt-get -y install python-pip git libopenblas-dev
  /usr/bin/yes | sudo pip install --upgrade google-api-python-client
  /usr/bin/yes | sudo pip install --upgrade oauth2client
  install_bazel
}

function install_and_setup_conda() {
  # Install conda if dne already.
  if ! test -d "$HOME/anaconda3"; then
    curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
    sh Anaconda3-5.2.0-Linux-x86_64.sh -b
    rm Anaconda3-5.2.0-Linux-x86_64.sh
  fi
  export PATH="$HOME/anaconda3/bin:$PATH"
  ENVNAME="pytorch"
  if conda env list | awk '{print $1}' | grep "^$ENVNAME$"; then
    conda remove --name "$ENVNAME" --all
  fi
  conda create -y --name "$ENVNAME" python=3.5 anaconda
  source activate "$ENVNAME"
  export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
  conda install -y numpy pyyaml setuptools cmake cffi typing
  sudo /sbin/ldconfig "${HOME}/anaconda3/lib/" "${HOME}/anaconda3/envs/pytorch/lib"
  /usr/bin/yes | pip install lark-parser
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
