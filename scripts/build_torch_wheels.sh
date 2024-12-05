#!/bin/bash

set -e  # Fail on any error.
set -x  # Display commands being run.

PYTHON_VERSION=$1
RELEASE_VERSION=$2  # rX.Y or nightly
BUILD_CPP_TESTS="${3:-0}"
DEFAULT_PYTHON_VERSION=3.8
DEBIAN_FRONTEND=noninteractive

ACL_VERSION=v22.05 # Arm Compute Library version

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
  if [[ $(uname -m) == "aarch64" && ! -d "$HOME/ComputeLibrary" ]]; then
    # install arm compute library
    pushd $HOME
    git clone https://review.mlplatform.org/ml/ComputeLibrary.git
    popd
  fi

  # Check if we have cloned pytorch or xla and cd into the pytorch dir.
  # Within the pytorch dir there is a subdir `torch`, and within xla
  # is `torch_xla`.
  if [ ! -d "torch" && ! -d "torch_xla" ]; then
    sudo apt-get install -y git
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    git clone --recursive https://github.com/pytorch/xla.git
    export RELEASE_VERSION="nightly"
  elif [ -d "torch_xla" ]; then
    cd ..
  fi
}

function install_bazel() {
  sudo apt-get install -y pkg-config zip zlib1g-dev unzip
  sudo apt-get -qq install npm nodejs
  sudo npm install -g @bazel/bazelisk
  if [[ -e /usr/bin/bazel ]]; then
    sudo unlink /usr/bin/bazel
  fi
  sudo ln -s "$(command -v bazelisk)" /usr/bin/bazel
  export PATH="$PATH:$HOME/bin"
}

function install_ninja() {
  sudo apt-get install ninja-build
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
  sudo update-alternatives --install /usr/bin/clang++ clang++ $(which clang++-8) 70
}

function install_gcc10() {
  sudo apt-get -y install gcc-10 g++-10
  export CC=/usr/bin/gcc-10 export CXX=/usr/bin/g++-10
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
}

function install_req_packages() {
  sudo apt-get -y install python3-pip git curl libopenblas-dev vim apt-transport-https ca-certificates wget procps
  maybe_install_cuda
  install_bazel
  install_ninja
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
  if [[ $(uname -m) == "x86_64" ]]; then
    if ! test -d "$HOME/anaconda3"; then
       CONDA_VERSION="5.3.1"
       curl -O "https://repo.anaconda.com/archive/Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"
       sh "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh" -b
       rm -f "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"
    fi
    maybe_append ". $HOME/anaconda3/etc/profile.d/conda.sh" ~/.bashrc
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ $(uname -m) == "aarch64" ]]; then
    if ! test -d "$HOME/miniforge3"; then
       curl -OL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
       sh -f Miniforge3-Linux-aarch64.sh -b
       rm -f Miniforge3-Linux-aarch64.sh
    fi
    maybe_append ". $HOME/miniforge3/etc/profile.d/conda.sh" ~/.bashrc
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  fi
  ENVNAME="pytorch"
  if conda env list | awk '{print $1}' | grep "^$ENVNAME$"; then
    conda remove -y --name "$ENVNAME" --all
  fi
  if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION=$DEFAULT_PYTHON_VERSION
  fi
  if [[ $(uname -m) == "x86_64" ]]; then
    conda create -y --name "$ENVNAME" python=${PYTHON_VERSION} anaconda
  elif [[ $(uname -m) == "aarch64" ]]; then
    conda create -y --name "$ENVNAME" python=${PYTHON_VERSION}
  fi

  conda activate "$ENVNAME"
  export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

  conda install -y numpy pyyaml setuptools==65.6.3 cmake cffi typing tqdm coverage tensorboard hypothesis dataclasses
  if [[ $(uname -m) == "x86_64" ]]; then
    # Overwrite mkl packages here, since nomkl conflicts with the anaconda env setup.
    conda remove -y tbb
    pip install mkl==2022.2.1 mkl_include==2022.2.1
  fi

  /usr/bin/yes | pip install --upgrade google-api-python-client
  /usr/bin/yes | pip install --upgrade oauth2client
  /usr/bin/yes | pip install --upgrade google-cloud-storage
  /usr/bin/yes | pip install lark-parser
  /usr/bin/yes | pip install cloud-tpu-client
  /usr/bin/yes | pip install expecttest==0.1.3
  /usr/bin/yes | pip install tensorboardX
}

function build_armcomputelibrary() {
  # setup additional system config, install scons
  sudo apt-get install -y scons
  if ! test -d "$HOME/acl"; then
     mkdir $HOME/acl
  fi
  install_dir=$HOME/acl

  # Build with scons
  scons -j16  Werror=0 debug=0 neon=1 opencl=0 embed_kernels=0 \
     openmp=1 cppthreads=0 os=linux arch=armv8.2-a build=native multi_isa=1 \
     build_dir=$install_dir/build

  cp -r arm_compute $install_dir
  cp -r include $install_dir
  cp -r utils $install_dir
  cp -r support $install_dir
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

  if [[ $(uname -m) == "aarch64" ]]; then
    # use mkldnn and acl backend for aarch64
    pushd $HOME/ComputeLibrary
    git checkout $ACL_VERSION
    build_armcomputelibrary
    popd
    export ACL_ROOT_DIR=$HOME/acl USE_OPENMP=1 USE_MKLDNN=ON USE_MKLDNN_ACL=ON
  fi

  python setup.py bdist_wheel
  pip install dist/*.whl

  # collect_wheels expects this
  if [ ! -d "/pytorch/dist" ]; then
    mkdir -p /pytorch/dist
    cp dist/*.whl /pytorch/dist/
  fi
}

function build_and_install_torch_xla() {
  git submodule update --init --recursive
  if [ "${RELEASE_VERSION}" = "nightly" ]; then
    export GIT_VERSIONED_XLA_BUILD=true
  else
    export TORCH_XLA_VERSION=${RELEASE_VERSION:1}  # r0.5 -> 0.5
  fi

  if [[ $(uname -m) == "aarch64" ]]; then
    # enable ACL runtime for CPU backend
    export XLA_CPU_USE_ACL=1
  fi

  python setup.py bdist_wheel
  pip install dist/*.whl
  if [ "$TPUVM_MODE" == "1" ]; then
    pip install torch_xla[tpu] \
      -f https://storage.googleapis.com/libtpu-wheels/index.html \
      -f https://storage.googleapis.com/libtpu-releases/index.html
    sudo apt-get install -y google-perftools
  fi

  # collect_wheels expects this
  if [ ! -d "/pytorch/xla/dist" ]; then
    mkdir -p /pytorch/xla/dist
    cp dist/*.whl /pytorch/xla/dist/
  fi
}

function install_torchvision_from_source() {
  torchvision_repo_version="main"
  # Cannot install torchvision package with PyTorch installation from source.
  # https://github.com/pytorch/vision/issues/967
  git clone -b "${torchvision_repo_version}" https://github.com/pytorch/vision.git
  pushd vision
  python setup.py bdist_wheel
  pip install dist/*.whl
  popd

  # collect_wheels expects this
  if [ ! -d "/pytorch/vision/dist" ]; then
    mkdir -p /pytorch/vision/dist
    cp vision/dist/*.whl /pytorch/vision/dist/
  fi
}

function install_torchaudio_from_source() {
  torchaudio_repo_version="main"
  git clone -b "${torchaudio_repo_version}" https://github.com/pytorch/audio.git
  pushd audio
  python setup.py bdist_wheel
  pip install dist/*.whl
  popd

  # collect_wheels expects this
  if [ ! -d "/pytorch/audio/dist" ]; then
    mkdir -p /pytorch/audio/dist
    cp audio/dist/*.whl /pytorch/audio/dist/
  fi
}

function main() {
  setup_system
  maybe_install_sources
  install_req_packages
  if [[ $(uname -m) == "x86_64" ]]; then
    install_llvm_clang
  elif [[ $(uname -m) == "aarch64" ]]; then
    install_gcc10
  fi
  install_and_setup_conda
  build_and_install_torch
  pushd xla
  build_and_install_torch_xla
  popd
  install_torchvision_from_source
  # install_torchaudio_from_source
  install_gcloud
}

main
