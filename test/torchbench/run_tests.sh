#!/bin/bash
set -ex
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
echo $CDIR

PYTORCH_DIR=$1
XLA_DIR=$2
TORCHBENCH_DIR=$PYTORCH_DIR/benchmark
shift 2
TORCHBENCH_MODELS=("$@")

# Note [Keep Going]
#
# Set the `CONTINUE_ON_ERROR` flag to `true` to make the CI tests continue on error.
# This will allow you to see all the failures on your PR, not stopping with the first
# test failure like the default behavior.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  set +e
fi

function install_torchbench_models() {
  pushd $CDIR
  cd ../../../benchmark/
  for model in "${TORCHBENCH_MODELS[@]}"; do
      echo "Installing model: $model"
      python install.py models "$model"
      if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install $model. Exiting." >&2
        exit 1
      fi
  done
  popd
}

success_count=0
function run_tests {
  local overall_status=0
  local pjrt_device="CPU"
  # TODO(piz): Uncomment the following if we decide to run on GPU.
  # if [ -x "$(command -v nvidia-smi)" ]; then
  #   num_devices=$(nvidia-smi --list-gpus | wc -l)
  #   echo "Found $num_devices GPU devices..."
  #   export GPU_NUM_DEVICES=$num_devices
  #   pjrt_device="CUDA"
  # fi
  for model in "${TORCHBENCH_MODELS[@]}"; do
    echo "testing model: $model"
    PJRT_DEVICE=$pjrt_device python -u benchmarks/experiment_runner.py \
    --suite-name=torchbench \
    --experiment-config='{"accelerator":"'"$pjrt_device"'","xla":"PJRT","dynamo":"openxla","test":"eval","torch_xla2":null,"xla_flags":null}' \
    --model-config='{"model_name":"'"$model"'"}'
    if [ $? -ne 0 ]; then
      echo "ERROR: Failed to test $model. Exiting with failure." >&2
      overall_status=1
    else
      success_count=$((success_count + 1))
    fi
  done
  return $overall_status
}


install_torchbench_models
run_tests
if [ $? -ne 0 ]; then
  echo "Torchbench test suite failed."
  exit 1
else
  echo "All torchbench tests passed successfully."
fi
total_models=${#TORCHBENCH_MODELS[@]}
echo "Successful tests: $success_count out of $total_models models"
