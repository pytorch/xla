set -ex

function run_torch_xla_python_tests() {
  XLA_DIR=$1
  USE_COVERAGE="${2:-0}"

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
  XLA_DIR=$1
  USE_COVERAGE="${2:-0}"

  TORCH_DIR=$(python -c "import pkgutil; import os; print(os.path.dirname(pkgutil.get_loader('torch').get_filename()))")
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${TORCH_DIR}/lib
  export PJRT_DEVICE=CPU
  export CPU_NUM_DEVICES=2
  export XLA_EXPERIMENTAL="nonzero:masked_select:nms"

  test_names=("test_aten_xla_tensor_1"
               "test_aten_xla_tensor_2"
               "test_aten_xla_tensor_3"
               "test_aten_xla_tensor_4"
               "pjrt_computation_client_test"
               # Disable IFRT test as it currently crashes
               #"ifrt_computation_client_test")
               "test_aten_xla_tensor_5"
               "test_aten_xla_tensor_6"
               "test_ir"
               "test_lazy"
               "test_replication"
               "test_tensor"
               # disable test_xla_backend_intf since it is flaky on upstream
               #"test_xla_backend_intf"
               "test_xla_generator"
               "test_xla_sharding"
               "test_runtime"
               "test_status_dont_show_cpp_stacktraces"
               "test_status_show_cpp_stacktraces"
               "test_debug_macros")
  for name in "${test_names[@]}"; do
    echo "Running $name cpp test..."
    /tmp/test/bin/${name}
  done
}

function run_torch_xla_benchmark_tests() {
  XLA_DIR=$1
  TORCHBENCH_MODELS=(BERT_pytorch dcgan)
  pushd $XLA_DIR
  echo "Running Benchmark Tests"
  test/benchmarks/run_tests.sh -L""
  popd
  pushd $XLA_DIR
  echo "Running Torchbench Tests"
  test/benchmarks/run_torchbench_tests.sh "${TORCHBENCH_MODELS[@]}"
  popd
}

PYTORCH_DIR=$1
XLA_DIR=$2
USE_COVERAGE="${3:-0}"

export PYTORCH_TESTING_DEVICE_ONLY_FOR="xla"
export CXX_ABI=$(python -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

if [[ -z "$RUN_BENCHMARK_TESTS" && -z "$RUN_CPP_TESTS" && -z "$RUN_PYTHON_TESTS" ]]; then
  run_torch_xla_python_tests $XLA_DIR $USE_COVERAGE
  run_torch_xla_cpp_tests $XLA_DIR $USE_COVERAGE
  run_torch_xla_benchmark_tests $XLA_DIR
else
  # run tests separately.
  if [[ "$RUN_PYTHON_TESTS" == "python_tests" ]]; then
    run_torch_xla_python_tests $XLA_DIR $USE_COVERAGE
  elif [[ "$RUN_BENCHMARK_TESTS" == "benchmark_tests" ]]; then
    run_torch_xla_benchmark_tests $XLA_DIR
  else
    run_torch_xla_cpp_tests $XLA_DIR $USE_COVERAGE
  fi
fi
