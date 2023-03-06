#!/bin/bash

set -ex

# See Note [Keep Going]
CONTINUE_ON_ERROR=false
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  set +e
fi

# System default cmake 3.10 cannot find mkl, so point it to the right place.
# CMAKE_PREFIX_PATH will default to (in this order):
# 1. CMAKE_PREFIX_PATH (if it exists)
# 2. CONDA_PREFIX (if it exists)
# 3. The conda install directory (if it exists)
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-${CONDA_PREFIX:-"$(dirname $(which conda))/../"}}

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
  USE_CACHE="${2:-0}"

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

  # Use cloud cache to build when available.
  if [[ "$USE_CACHE" == 1 ]]; then
    # Install bazels3cache for cloud cache
    sudo npm install -g n
    sudo n 16.18.0
    sudo npm install -g bazels3cache
    BAZELS3CACHE="$(which /usr/local/bin/bazels3cache)"
    if [ -z "${BAZELS3CACHE}" ]; then
      echo "Unable to find bazels3cache..."
      exit 1
    fi
    /usr/local/bin/bazels3cache --bucket=${XLA_CLANG_CACHE_S3_BUCKET_NAME} --maxEntrySizeBytes=0 --logging.level=verbose
    sed -i '/bazel build/ a --remote_http_cache=http://localhost:7777 \\' $XLA_DIR/build_torch_xla_libs.sh
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
  USE_COVERAGE="${3:-0}"
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
    if [ "$USE_COVERAGE" != "0" ]; then
      pip install coverage==6.5.0 --upgrade
      ./test/utils/run_test_coverage.sh
      coverage combine
      coverage html
      mv htmlcov ~/
      chmod -R 755 ~/htmlcov
    else
      ./test/run_tests.sh
      # only run test_autocast for cpu and gpu on circleCI.
      python test/test_autocast.py

    # GPU tests
    if [ -x "$(command -v nvidia-smi)" ]; then
      python test/test_train_mp_imagenet_fsdp.py --fake_data --use_nested_fsdp --use_small_fake_sample --num_epochs=1
      python test/test_train_mp_imagenet_fsdp.py --fake_data --auto_wrap_policy type_based --use_small_fake_sample --num_epochs=1
      # Syncfree SGD optimizer tests
      if [ -d ./torch_xla/amp/syncfree ]; then
        echo "Running Syncfree Optimizer Test"
        python test/test_syncfree_optimizers.py

        # Following test scripts are mainly useful for
        # performance evaluation & comparison among different
        # amp optimizers.
        # echo "Running ImageNet Test"
        # python test/test_train_mp_imagenet_amp.py --fake_data --num_epochs=1

        # disabled per https://github.com/pytorch/xla/pull/2809
        # echo "Running MNIST Test"
        # python test/test_train_mp_mnist_amp.py --fake_data --num_epochs=1
      fi
    fi

    pushd test/cpp
    echo "Running C++ Tests on PJRT"
    if [ -x "$(command -v nvidia-smi)" ]; then
      PJRT_DEVICE=GPU ./run_tests.sh
    else
      PJRT_DEVICE=CPU ./run_tests.sh
    fi

      if ! [ -x "$(command -v nvidia-smi)"  ]
      then
        ./run_tests.sh -X early_sync -F AtenXlaTensorTest.TestEarlySyncLiveTensors -L""
      fi
      popd
    fi
  popd
}
