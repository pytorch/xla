#!/bin/bash

set -e  # Fail on any error.
set -x  # Display commands being run.

PYTHON_VERSION=$1
RELEASE_VERSION=$2  # rX.Y or nightly
DEFAULT_PYTHON_VERSION=3.6
DEBIAN_FRONTEND=noninteractive

function install_bazel() {
  local BAZEL_VERSION="1.1.0"
  local BAZEL_FILE="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
  sudo apt-get install -y pkg-config zip zlib1g-dev unzip
  curl -L -O "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${BAZEL_FILE}"
  chmod 755 "$BAZEL_FILE"
  ./"$BAZEL_FILE" --user
  export PATH="$PATH:$HOME/bin"
}

function install_llvm_clang() {
  sudo apt-get -y install wget
  sudo bash -c 'echo "deb http://apt.llvm.org/stretch/ llvm-toolchain-stretch-7 main" >> /etc/apt/sources.list'
  sudo bash -c 'echo "deb-src http://apt.llvm.org/stretch/ llvm-toolchain-stretch-7 main" >> /etc/apt/sources.list'
  wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
  sudo apt-get update
  sudo apt-get -y install clang-7 clang++-7
  echo 'export CC=clang-7 CXX=clang++-7' >> ~/.bashrc
  export CC=clang-7 CXX=clang++-7
}

function install_req_packages() {
  sudo apt-get -y install python-pip git curl libopenblas-dev vim
  install_bazel
}

function install_gcloud() {
  # Following: https://cloud.google.com/sdk/docs/downloads-apt-get
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
  sudo apt-get install -y apt-transport-https ca-certificates
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
  sudo apt-get update && sudo apt-get install -y google-cloud-sdk
}

function install_and_setup_conda() {
  # Install conda if dne already.
  if ! test -d "$HOME/anaconda3"; then
    curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
    sh Anaconda3-5.2.0-Linux-x86_64.sh -b
    rm Anaconda3-5.2.0-Linux-x86_64.sh
  fi
  echo ". $HOME/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  ENVNAME="pytorch"
  if conda env list | awk '{print $1}' | grep "^$ENVNAME$"; then
    conda remove --name "$ENVNAME" --all
  fi
  if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION=$DEFAULT_PYTHON_VERSION
  fi
  conda create -y --name "$ENVNAME" python=${PYTHON_VERSION} anaconda
  conda activate "$ENVNAME"
  export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

  conda install -y numpy pyyaml setuptools cmake cffi typing tqdm coverage tensorboard hypothesis
  /usr/bin/yes | pip install --upgrade google-api-python-client
  /usr/bin/yes | pip install --upgrade oauth2client
  /usr/bin/yes | pip install --upgrade google-cloud-storage
  /usr/bin/yes | pip install lark-parser
  /usr/bin/yes | pip install cloud-tpu-client
  /usr/bin/yes | pip install tensorboardX

  sudo /sbin/ldconfig "${HOME}/anaconda3/lib/" "${HOME}/anaconda3/envs/pytorch/lib"
}

function build_and_install_torch() {
  # Checkout the PT commit ID if we have one.
  COMMITID_FILE="xla/.torch_commit_id"
  if [ -e "$COMMITID_FILE" ]; then
    git checkout $(cat "$COMMITID_FILE")
  fi
  # Only checkout dependencies once PT commit ID checked out.
  git submodule update --init --recursive
  # Apply patches to PT which are required by the XLA support.
  $(dirname $0)/apply_patches.sh
  export NO_CUDA=1
  python setup.py bdist_wheel
  pip install dist/*.whl
}

function build_and_install_torch_xla() {
  git submodule update --init --recursive
  export NO_CUDA=1
  if [ "${RELEASE_VERSION}" = "nightly" ]; then
    export VERSIONED_XLA_BUILD=1
  else
    export TORCH_XLA_VERSION=${RELEASE_VERSION:1}  # r0.5 -> 0.5
  fi
  python setup.py bdist_wheel
  pip install dist/*.whl
}

function install_torchvision_from_source() {
  torchvision_repo_version="master"
  # Cannot install torchvision package with PyTorch installation from source.
  # https://github.com/pytorch/vision/issues/967
  git clone -b "${torchvision_repo_version}" https://github.com/pytorch/vision.git
  pushd vision
  python setup.py bdist_wheel
  pip install dist/*.whl
  popd
}

function main() {
  install_llvm_clang
  install_req_packages
  install_and_setup_conda
  build_and_install_torch
  pushd xla
  build_and_install_torch_xla
  popd
  install_torchvision_from_source
  install_gcloud
}

main
