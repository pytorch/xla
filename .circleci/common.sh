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

function install_pre_deps_pytorch_xla() {
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
  pip install pandas
  pip install tabulate
  pip install --upgrade "numpy>=1.18.5"
  pip install --upgrade numba

  # Using the Ninja generator requires CMake version 3.13 or greater
  pip install "cmake>=3.20" --upgrade

  sudo apt-get -qq update

  sudo apt-get -qq install npm nodejs

  # Install LCOV and llvm-cov to generate C++ coverage reports
  sudo apt-get install -y lcov

  # XLA build requires Bazel
  # We use bazelisk to avoid updating Bazel version manually.
  # Check and remove any existing bazel symlinks first
  for bazel_path in /usr/bin/bazel /usr/local/bin/bazel; do
    if [[ -e "$bazel_path" ]]; then
      sudo rm -f "$bazel_path"
    fi
  done

  # Install bazelisk
  sudo npm install -g @bazel/bazelisk

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
  # Need to uncomment the line below.
  # Currently it fails upstream XLA CI.
  # pip install plugins/cuda -v
  popd
}

function run_torch_xla_python_tests() {
  PYTORCH_DIR=$1
  XLA_DIR=$2
  USE_COVERAGE="${3:-0}"

  pushd $XLA_DIR
    echo "Running Python Tests"
    if [ "$USE_COVERAGE" != "0" ]; then
      pip install coverage==6.5.0 --upgrade
      pip install coverage-lcov
      pip install toml
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
    fi
  popd
}

function run_torch_xla_cpp_tests() {
  PYTORCH_DIR=$1
  XLA_DIR=$2
  USE_COVERAGE="${3:-0}"

  pushd $XLA_DIR
    echo "Running C++ Tests on PJRT"
    EXTRA_ARGS=""
    if [ "$USE_COVERAGE" != "0" ]; then
	    EXTRA_ARGS="-C"
    fi
    if [ ! -z "$GCLOUD_SERVICE_KEY_FILE" ]; then
	    EXTRA_ARGS="$EXTRA_ARGS -R"
    fi

    if [ "$USE_COVERAGE" != "0" ]; then
      if [ -x "$(command -v nvidia-smi)" ]; then
        PJRT_DEVICE=CUDA test/cpp/run_tests.sh $EXTRA_ARGS -L""
        cp $XLA_DIR/bazel-out/_coverage/_coverage_report.dat /tmp/cov1.dat
        PJRT_DEVICE=CUDA test/cpp/run_tests.sh -X early_sync -F AtenXlaTensorTest.TestEarlySyncLiveTensors -L"" $EXTRA_ARGS
        cp $XLA_DIR/bazel-out/_coverage/_coverage_report.dat /tmp/cov2.dat
        lcov --add-tracefile /tmp/cov1.dat -a /tmp/cov2.dat -o /tmp/merged.dat
      else
        PJRT_DEVICE=CPU test/cpp/run_tests.sh $EXTRA_ARGS -L""
        cp $XLA_DIR/bazel-out/_coverage/_coverage_report.dat /tmp/merged.dat
      fi
      genhtml /tmp/merged.dat -o ~/htmlcov/cpp/cpp_lcov.info
      mv /tmp/merged.dat ~/htmlcov/cpp_lcov.info
    else
      # Shard GPU testing
      if [ -x "$(command -v nvidia-smi)" ]; then
        PJRT_DEVICE=CUDA test/cpp/run_tests.sh $EXTRA_ARGS -L""
        PJRT_DEVICE=CUDA test/cpp/run_tests.sh -X early_sync -F AtenXlaTensorTest.TestEarlySyncLiveTensors -L"" $EXTRA_ARGS
      else
        PJRT_DEVICE=CPU test/cpp/run_tests.sh $EXTRA_ARGS -L""
      fi
    fi
  popd
}

function run_torch_xla_benchmark_tests() {
  XLA_DIR=$1
  pushd $XLA_DIR
    echo "Running Benchmark Tests"
    test/benchmarks/run_tests.sh -L""
}

function run_torch_xla_tests() {
  PYTORCH_DIR=$1
  XLA_DIR=$2
  USE_COVERAGE="${3:-0}"
  RUN_CPP="${RUN_CPP_TESTS:0}"
  RUN_PYTHON="${RUN_PYTHON_TESTS:0}"

  if [ -x "$(command -v nvidia-smi)" ]; then
    num_devices=$(nvidia-smi --list-gpus | wc -l)
    echo "Found $num_devices GPU devices..."
    export GPU_NUM_DEVICES=$num_devices
  fi
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="xla"
  export CXX_ABI=$(python -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

  if [[ -z "$RUN_BENCHMARK_TESTS" && -z "$RUN_CPP_TESTS" && -z "$RUN_PYTHON_TESTS" ]]; then
    run_torch_xla_python_tests $PYTORCH_DIR $XLA_DIR $USE_COVERAGE
    run_torch_xla_cpp_tests $PYTORCH_DIR $XLA_DIR $USE_COVERAGE
    run_torch_xla_benchmark_tests $XLA_DIR
  else
    # run tests separately.
    if [[ "$RUN_PYTHON_TESTS" == "python_tests" ]]; then
      run_torch_xla_python_tests $PYTORCH_DIR $XLA_DIR $USE_COVERAGE
    elif [[ "$RUN_BENCHMARK_TESTS" == "benchmark_tests" ]]; then
      run_torch_xla_benchmark_tests $XLA_DIR
    else
      run_torch_xla_cpp_tests $PYTORCH_DIR $XLA_DIR $USE_COVERAGE
    fi
  fi
}
