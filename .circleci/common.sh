#!/bin/bash

set -ex

# System default cmake 3.10 cannot find mkl, so point it to the right place.
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

function clone_pytorch() {
  PYTORCH_DIR=$1
  XLA_DIR=$2
  git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
  cp -r "$PWD" "$XLA_DIR"
}

function apply_patches() {
  # assumes inside pytorch dir
  ./xla/scripts/apply_patches.sh
}

function rebase_pull_request_on_target_branch() {
  # TODO: directly use ENV_VAR when CircleCi exposes base branch.
  # Try rebasing on top of base (dest) branch first.
  # This allows us to pickup the latest fix for PT-XLA breakage.
  # Also it might improve build time as we have warm cache.
  git config --global user.email "circleci.ossci@gmail.com"
  git config --global user.name "CircleCI"
  sudo apt-get update && sudo apt-get -qq install jq
  # Only rebase on runs triggered by PR checks not post-submits.
  if [[ ! -z "${CIRCLE_PULL_REQUEST}" ]]; then
    PR_NUM=$(basename $CIRCLE_PULL_REQUEST)
    CIRCLE_PR_BASE_BRANCH=$(curl -s https://api.github.com/repos/$CIRCLE_PROJECT_USERNAME/$CIRCLE_PROJECT_REPONAME/pulls/$PR_NUM | jq -r '.base.ref')
    git rebase "origin/${CIRCLE_PR_BASE_BRANCH}"
    git submodule deinit -f .
    git submodule update --init --recursive
  fi
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
  # Install ninja to speedup the build
  pip install ninja

  # Install libraries required for running some PyTorch test suites
  pip install hypothesis
  pip install cloud-tpu-client

  # Bazel doesn't work with sccache gcc. https://github.com/bazelbuild/bazel/issues/3642
  sudo apt-get -qq update

  sudo apt-get -qq install npm nodejs

  # XLA build requires Bazel
  # We use bazelisk to avoid updating Bazel version manually.
  sudo npm install -g @bazel/bazelisk
  sudo ln -s "$(command -v bazelisk)" /usr/bin/bazel

  # Install bazels3cache for cloud cache
  sudo npm install -g bazels3cache
  BAZELS3CACHE="$(which bazels3cache)"
  if [ -z "${BAZELS3CACHE}" ]; then
    echo "Unable to find bazels3cache..."
    exit 1
  fi
  bazels3cache --bucket=${XLA_CLANG_CACHE_S3_BUCKET_NAME} --maxEntrySizeBytes=0 --logging.level=verbose
  # Use cloud cache to build when available.
  sed -i '/bazel build/ a --remote_http_cache=http://localhost:7777 \\' $XLA_DIR/build_torch_xla_libs.sh
}

function build_torch_xla() {
  XLA_DIR=$1
  pushd "$XLA_DIR"
  CC=clang-9 CXX=clang++-9 python setup.py install
  popd
}

function run_torch_xla_tests() {
  PYTORCH_DIR=$1
  XLA_DIR=$2
  if [ -x "$(command -v nvidia-smi)" ]; then
    export GPU_NUM_DEVICES=2
  else
    export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
    XLA_PORT=$(shuf -i 40701-40999 -n 1)
    export XRT_WORKERS="localservice:0;grpc://localhost:$XLA_PORT"
  fi
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="xla"

  pushd $XLA_DIR
    echo "Running Python Tests"
    ./test/run_tests.sh

    # echo "Running MNIST Test"
    # python test/test_train_mp_mnist.py --tidy
    if [ -x "$(command -v nvidia-smi)" ]; then
      python test/test_autocast.py
      # python test/test_train_mp_mnist_amp.py --fake_data
    fi

    pushd test/cpp
    echo "Running C++ Tests"
    CC=clang-9 CXX=clang++-9 ./run_tests.sh

    if ! [ -x "$(command -v nvidia-smi)"  ]
    then
      CC=clang-9 CXX=clang++-9 ./run_tests.sh -X early_sync -F AtenXlaTensorTest.TestEarlySyncLiveTensors -L""
    fi
    popd
  popd
}
