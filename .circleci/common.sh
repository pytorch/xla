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
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python3-config --prefix)/lib"
echo $LD_LIBRARY_PATH

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

  # Hack similar to https://github.com/pytorch/pytorch/pull/105227/files#diff-9e59213240d3b55d2ddc53c8c096db9eece0665d64f46473454f9dc0c10fd804
  # TODO(yeounoh) confirm that this is not needed here.
  sudo rm /opt/conda/lib/libstdc++.so*

  sudo apt-get -qq install npm nodejs

  # Install LCOV and llvm-cov to generate C++ coverage reports
  sudo apt-get install -y lcov

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
  USE_COVERAGE="${3:-0}"
  if [ -x "$(command -v nvidia-smi)" ]; then
    export GPU_NUM_DEVICES=2
  fi
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="xla"
  export CXX_ABI=$(python -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

  pushd $XLA_DIR
    echo "Running Python Tests"
    if [ "$USE_COVERAGE" != "0" ]; then
      pip install coverage==6.5.0 --upgrade
      pip install coverage-lcov
      ./test/run_tests.sh
      coverage combine
      mkdir lcov && cp .coverage lcov/
      coverage-lcov --data_file_path lcov/.coverage
      coverage html
      cp lcov.info htmlcov/
      mv htmlcov ~/
      chmod -R 755 ~/htmlcov
    else
      ./test/run_tests.sh

      # GPU tests
      if [ -x "$(command -v nvidia-smi)" ]; then
        # These tests fail on GPU with 03/30 TF-pin update (https://github.com/pytorch/xla/pull/4840)
        # PJRT_DEVICE=GPU python test/test_train_mp_imagenet_fsdp.py --fake_data --use_nested_fsdp --use_small_fake_sample --num_epochs=1
        # PJRT_DEVICE=GPU python test/test_train_mp_imagenet_fsdp.py --fake_data --auto_wrap_policy type_based --use_small_fake_sample --num_epochs=1
        # XLA_DISABLE_FUNCTIONALIZATION=1 PJRT_DEVICE=GPU python test/test_train_mp_imagenet_fsdp.py --fake_data --use_nested_fsdp --use_small_fake_sample --num_epochs=1
        # Syncfree SGD optimizer tests
        if [ -d ./torch_xla/amp/syncfree ]; then
          echo "Running Syncfree Optimizer Test"
          PJRT_DEVICE=GPU python test/test_syncfree_optimizers.py

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
    fi

    echo "Running C++ Tests on PJRT"
    EXTRA_ARGS=""
    if [ "$USE_COVERAGE" != "0" ]; then
	    EXTRA_ARGS="-C"
    fi
    if [ ! -z "$GCLOUD_SERVICE_KEY_FILE" ]; then
	    EXTRA_ARGS="$EXTRA_ARGS -R"
    fi
    if [ -x "$(command -v nvidia-smi)" ]; then
      PJRT_DEVICE=GPU test/cpp/run_tests.sh $EXTRA_ARGS -L""
      if [ "$USE_COVERAGE" != "0" ]; then
        cp $XLA_DIR/bazel-out/_coverage/_coverage_report.dat /tmp/cov1.dat
      fi
      PJRT_DEVICE=GPU test/cpp/run_tests.sh -X early_sync -F AtenXlaTensorTest.TestEarlySyncLiveTensors -L"" $EXTRA_ARGS
      if [ "$USE_COVERAGE" != "0" ]; then
        cp $XLA_DIR/bazel-out/_coverage/_coverage_report.dat /tmp/cov2.dat
        lcov --add-tracefile /tmp/cov1.dat -a /tmp/cov2.dat -o /tmp/merged.dat
      fi
    else
      PJRT_DEVICE=CPU test/cpp/run_tests.sh $EXTRA_ARGS -L""
      if [ "$USE_COVERAGE" != "0" ]; then
        cp $XLA_DIR/bazel-out/_coverage/_coverage_report.dat /tmp/merged.dat
      fi
    fi
    if [ "$USE_COVERAGE" != "0" ]; then
      genhtml /tmp/merged.dat -o ~/htmlcov/cpp/cpp_lcov.info
      mv /tmp/merged.dat ~/htmlcov/cpp_lcov.info
    fi
  popd
}
