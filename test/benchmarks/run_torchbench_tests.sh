#!/bin/bash
set -ex
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
echo $CDIR

TORCHBENCH_MODELS=("$@")
# construct the absolute path
XLA_DIR=$CDIR/../../
PYTORCH_DIR=$XLA_DIR/../
TORCHVISION_DIR=$PYTORCH_DIR/vision
TORCHAUDIO_DIR=$PYTORCH_DIR/audio
TORCHTEXT_DIR=$PYTORCH_DIR/text
TORCHBENCH_DIR=$PYTORCH_DIR/benchmark

# Note [Keep Going]
#
# Set the `CONTINUE_ON_ERROR` flag to `true` to make the CI tests continue on error.
# This will allow you to see all the failures on your PR, not stopping with the first
# test failure like the default behavior.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  set +e
fi


function install_package() {
  pushd $CDIR

  torchvision_commit_hash=$(cat $PYTORCH_DIR/.github/ci_commit_pins/vision.txt)
  echo torchvision_commit_hash: "$torchvision_commit_hash"
  git clone --quiet https://github.com/pytorch/vision.git "$TORCHVISION_DIR"
  cd $TORCHVISION_DIR
  git checkout $torchvision_commit_hash
  python setup.py install 1>/dev/null

  torchaudio_commit_hash=$(cat $PYTORCH_DIR/.github/ci_commit_pins/audio.txt)
  echo torchaudio_commit_hash: "$torchaudio_commit_hash"
  git clone --quiet https://github.com/pytorch/audio.git "$TORCHAUDIO_DIR"
  cd $TORCHAUDIO_DIR
  git checkout $torchaudio_commit_hash
  python setup.py install 1>/dev/null

  torchtext_commit_hash=$(cat $PYTORCH_DIR/.github/ci_commit_pins/text.txt)
  echo torchtext_commit_hash: "$torchtext_commit_hash"
  git clone --quiet https://github.com/pytorch/text.git "$TORCHTEXT_DIR"
  cd $TORCHTEXT_DIR
  git checkout $torchtext_commit_hash
  git submodule update --init --recursive
  python setup.py clean install 1>/dev/null

  popd
}

function install_torchbench_models() {
  pushd $CDIR

  git clone --quiet https://github.com/pytorch/benchmark.git "$TORCHBENCH_DIR"
  cd $TORCHBENCH_DIR
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
    --experiment-config='{"accelerator":"'"$pjrt_device"'","xla":"PJRT","dynamo":"openxla","test":"eval","torch_xla2":null,"xla_flags":null,"keep_model_data_on_cuda":false}' \
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

install_package
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
