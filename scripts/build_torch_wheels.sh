#!/bin/bash

set -e  # Fail on any error.
set -x  # Display commands being run.

PYTHON_VERSION=$1
RELEASE_VERSION=$2  # rX.Y or nightly
DEFAULT_PYTHON_VERSION=3.6
DEBIAN_FRONTEND=noninteractive

function maybe_append {
  local LINE="$1"
  local FILE="$2"
  if [ ! -r "$FILE" ]; then
    if [ -w "$(dirname $FILE)" ]; then
      echo "$LINE" > "$FILE"
    else
      sudo bash -c "echo '$LINE' > \"$FILE\""
    fi
  elif [ "$(grep -F "$LINE" $FILE)" == "" ]; then
    if [ -w "$FILE" ]; then
      echo "$LINE" >> "$FILE"
    else
      sudo bash -c "echo '$LINE' >> \"$FILE\""
    fi
  fi
}

function setup_system {
  maybe_append 'APT::Acquire::Retries "10";' /etc/apt/apt.conf.d/80-failparams
  maybe_append 'APT::Acquire::http::Timeout "180";' /etc/apt/apt.conf.d/80-failparams
  maybe_append 'APT::Acquire::ftp::Timeout "180";' /etc/apt/apt.conf.d/80-failparams

  if [ "$CXX_ABI" == "0" ]; then
    export CFLAGS="${CFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0"
    export CXXFLAGS="${CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0"
  fi
}

function install_cudnn {
  if [ "$CUDNN_TGZ_PATH" == "" ]; then
    echo "Missing CUDNN_TGZ_PATH environment variable"
    exit 1
  fi
  local CUDNN_FILE="cudnn.tar.gz"
  if [[ "$CUDNN_TGZ_PATH" == gs://* ]]; then
    gsutil cp "$CUDNN_TGZ_PATH" "$CUDNN_FILE"
  elif [[ "$CUDNN_TGZ_PATH" == http* ]]; then
    wget -O "$CUDNN_FILE" "$CUDNN_TGZ_PATH"
  else
    ln -s "$CUDNN_TGZ_PATH" "$CUDNN_FILE"
  fi
  tar xvf "$CUDNN_FILE"
  sudo cp cuda/include/cudnn.h /usr/local/cuda/include
  sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
  sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
  rm -rf cuda
  rm -f "$CUDNN_FILE"
}

function maybe_install_cuda {
  if [ "$XLA_CUDA" == "1" ]; then
    if [ ! -d "/usr/local/cuda" ]; then
      local CUDA_VER="10.2"
      local CUDA_SUBVER="89_440.33.01"
      local CUDA_FILE="cuda_${CUDA_VER}.${CUDA_SUBVER}_linux.run"
      wget "http://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/Prod/local_installers/${CUDA_FILE}"
      sudo sh "${CUDA_FILE}" --silent --toolkit
      rm -f "${CUDA_FILE}"
    fi
    if [ ! -f "/usr/local/cuda/include/cudnn.h" ] && [ ! -f "/usr/include/cudnn.h" ]; then
      install_cudnn
    fi
    export TF_CUDA_PATHS="/usr/local/cuda,/usr/include,/usr"
    maybe_append 'export TF_CUDA_PATHS="/usr/local/cuda,/usr/include,/usr"' ~/.bashrc
    if [ "$TF_CUDA_COMPUTE_CAPABILITIES" == "" ]; then
      export TF_CUDA_COMPUTE_CAPABILITIES="7.0"
    fi
    maybe_append "export TF_CUDA_COMPUTE_CAPABILITIES=\"$TF_CUDA_COMPUTE_CAPABILITIES\"" ~/.bashrc
  fi
}

function maybe_install_sources {
  if [ ! -d "torch" ]; then
    sudo apt-get install -y git
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    git clone --recursive https://github.com/pytorch/xla.git
    export RELEASE_VERSION="nightly"
  fi
}

function install_bazel() {
  local BAZEL_VERSION=$(cat "xla/third_party/tensorflow/.bazelversion")
  local BAZEL_FILE="bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh"
  sudo apt-get install -y pkg-config zip zlib1g-dev unzip
  # Move to /tmp folder as otherwise the .bazelversion file found in pytorch source makes
  # the install fail.
  pushd /tmp
  curl -L -O "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/${BAZEL_FILE}"
  chmod 755 "$BAZEL_FILE"
  ./"$BAZEL_FILE" --user
  rm -f "$BAZEL_FILE"
  popd
  export PATH="$PATH:$HOME/bin"
}

function debian_version {
  local VER
  if ! sudo apt-get install -y lsb-release > /dev/null 2>&1 ; then
    VER="buster"
  else
    VER=$(lsb_release -c -s)
  fi
  echo "$VER"
}

function install_llvm_clang() {
  local DEBVER=$(debian_version)
  if ! apt-get install -y -s clang-8 > /dev/null 2>&1 ; then
    maybe_append "deb http://apt.llvm.org/${DEBVER}/ llvm-toolchain-${DEBVER}-8 main" /etc/apt/sources.list
    maybe_append "deb-src http://apt.llvm.org/${DEBVER}/ llvm-toolchain-${DEBVER}-8 main" /etc/apt/sources.list
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
    sudo apt-get update
  fi
  sudo apt-get install -y clang-8 clang++-8
  maybe_append 'export CC=clang-8 CXX=clang++-8' ~/.bashrc
  export CC=clang-8 CXX=clang++-8
}

function install_req_packages() {
  sudo apt-get -y install python-pip git curl libopenblas-dev vim apt-transport-https ca-certificates wget procps
  maybe_install_cuda
  install_bazel
}

function install_gcloud() {
  if [ "$(which gcloud)" == "" ]; then
    # Following: https://cloud.google.com/sdk/docs/downloads-apt-get
    maybe_append "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" /etc/apt/sources.list.d/google-cloud-sdk.list
    sudo apt-get install -y apt-transport-https ca-certificates
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
    sudo apt-get update && sudo apt-get install -y google-cloud-sdk
  fi
}

function install_and_setup_conda() {
  # Install conda if dne already.
  if ! test -d "$HOME/anaconda3"; then
    CONDA_VERSION="5.3.1"
    curl -O "https://repo.anaconda.com/archive/Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"
    sh "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh" -b
    rm -f "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"
  fi
  maybe_append ". $HOME/anaconda3/etc/profile.d/conda.sh" ~/.bashrc
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

  conda install -y numpy pyyaml mkl-include setuptools cmake cffi typing tqdm coverage tensorboard hypothesis dataclasses
  /usr/bin/yes | pip install --upgrade google-api-python-client
  /usr/bin/yes | pip install --upgrade oauth2client
  /usr/bin/yes | pip install --upgrade google-cloud-storage
  /usr/bin/yes | pip install lark-parser
  /usr/bin/yes | pip install cloud-tpu-client
  /usr/bin/yes | pip install tensorboardX
}

function build_and_install_torch() {
  # Checkout the PT commit ID or branch if we have one.
  COMMITID_FILE="xla/.torch_pin"
  if [ -e "$COMMITID_FILE" ]; then
    git checkout $(cat "$COMMITID_FILE")
  fi
  # Only checkout dependencies once PT commit/branch checked out.
  git submodule update --init --recursive
  # Apply patches to PT which are required by the XLA support.
  xla/scripts/apply_patches.sh
  python setup.py bdist_wheel
  pip install dist/*.whl
}

function build_and_install_torch_xla() {
  git submodule update --init --recursive
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
  setup_system
  maybe_install_sources
  install_req_packages
  install_llvm_clang
  install_and_setup_conda
  build_and_install_torch
  pushd xla
  build_and_install_torch_xla
  popd
  install_torchvision_from_source
  install_gcloud
}

main
