#!/bin/bash

set -ex

DEBIAN_FRONTEND=noninteractive

PYTORCH_DIR="${KOKORO_ARTIFACTS_DIR}/github/pytorch"
XLA_DIR=$PYTORCH_DIR/xla
git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
cp -r "${KOKORO_ARTIFACTS_DIR}/github/xla" "$XLA_DIR"
source ${XLA_DIR}/.kokoro/common.sh

TORCHVISION_COMMIT="$(cat $PYTORCH_DIR/.github/ci_commit_pins/vision.txt)"

install_environments

pip install --user https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl \
  https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torchvision-nightly-cp38-cp38-linux_x86_64.whl \
  https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
pip install torch_xla[tpuvm] --user

set +e
# -E is for preserving environment
run_torch_xla_tests $PYTORCH_DIR $XLA_DIR
test_status="$(echo $?)"
set -e

# Create mapping for flakiness reporting
CUSTOM_SPONGE_CONFIG="${KOKORO_ARTIFACTS_DIR}/custom_sponge_config.csv"

export_to_sponge_config() {
  local key="${1}"
  local value="${2}"
  printf "${key},${value}\n" >> "${CUSTOM_SPONGE_CONFIG}"
}

flakiness()
{
  pushd $XLA_DIR
  export_to_sponge_config "ng3_commit" "$(git rev-parse HEAD)"
  popd
  export_to_sponge_config "ng3_cl_target_branch" "master"
  export_to_sponge_config "ng3_project_id" "pytorchxla"
  export_to_sponge_config "ng3_job_type" "POSTSUBMIT"
  export_to_sponge_config "ng3_test_type" "UNIT"
}

flakiness || true

exit "${test_status}"


