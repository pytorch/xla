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
  if [ -x "$(command -v nvidia-smi)" ]; then
    CUDA_PLUGIN_DIR=$(python -c "import pkgutil; import os; print(os.path.dirname(pkgutil.get_loader('torch_xla_cuda_plugin').get_filename()))")
    export PJRT_LIBRARY_PATH=$CUDA_PLUGIN_DIR/lib/pjrt_c_api_gpu_plugin.so
    export PJRT_DEVICE=LIBRARY
    export PJRT_DYNAMIC_PLUGINS=1
  else
    export PJRT_DEVICE=CPU
  fi
  export XLA_EXPERIMENTAL="nonzero:masked_select:nms"

  test_names1=("test_aten_xla_tensor_1"
               "test_aten_xla_tensor_2"
               "test_aten_xla_tensor_3"
               "test_aten_xla_tensor_4"
               "pjrt_computation_client_test")
               # Disable IFRT test as it currently crashes
               #"ifrt_computation_client_test")
  test_names2=("test_aten_xla_tensor_5"
               "test_aten_xla_tensor_6"
               "test_ir"
               "test_lazy"
               "test_replication"
               "test_tensor"
               # disable test_xla_backend_intf since it is flaky on upstream
               #"test_xla_backend_intf"
               "test_xla_sharding")
  if [[ "$RUN_CPP_TESTS1" == "cpp_tests1" ]]; then
    test_names=("${test_names1[@]}")
  elif [[ "$RUN_CPP_TESTS2" == "cpp_tests2" ]]; then
    test_names=("${test_names2[@]}")
  else
    test_names=("${test_names1[@]}" "${test_names2[@]}")
  fi

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

if [ -x "$(command -v nvidia-smi)" ]; then
  num_devices=$(nvidia-smi --list-gpus | wc -l)
  echo "Found $num_devices GPU devices..."
  export GPU_NUM_DEVICES=$num_devices
fi
export PYTORCH_TESTING_DEVICE_ONLY_FOR="xla"
export CXX_ABI=$(python -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

if [[ -z "$RUN_BENCHMARK_TESTS" && -z "$RUN_CPP_TESTS1" && -z "$RUN_CPP_TESTS2" && -z "$RUN_PYTHON_TESTS" ]]; then
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
