#!/bin/bash

set -ex

DEBIAN_FRONTEND=noninteractive

PYTORCH_DIR="${KOKORO_ARTIFACTS_DIR}/github/pytorch"
XLA_DIR=$PYTORCH_DIR/xla
git clone --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
cp -r "${KOKORO_ARTIFACTS_DIR}/github/xla" "$XLA_DIR"
source ${XLA_DIR}/.kokoro/common.sh

TORCHVISION_COMMIT="$(cat $PYTORCH_DIR/.github/ci_commit_pins/vision.txt)"

apt-get clean && apt-get update
apt-get upgrade -y
apt-get install --fix-missing -y python3-pip git curl libopenblas-dev vim jq \
  apt-transport-https ca-certificates procps openssl sudo wget libssl-dev libc6-dbg

curl -LO https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
mv bazelisk-linux-amd64 /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

pip install mkl mkl-include setuptools typing_extensions cmake requests
sudo ln -s /usr/local/lib/libmkl_intel_lp64.so.2 /usr/local/lib/libmkl_intel_lp64.so.1
sudo ln -s /usr/local/lib/libmkl_intel_thread.so.2 /usr/local/lib/libmkl_intel_thread.so.1
sudo ln -s /usr/local/lib/libmkl_core.so.2 /usr/local/lib/libmkl_core.so.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" >> /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# TODO(yeounoh) fix `GoogleCredentials` import error
apt-get update
apt-get -y install google-cloud-cli
pip install --upgrade google-api-python-client
pip install --upgrade oauth2client
pip install --upgrade google-cloud-storage
pip install lark-parser
pip install cloud-tpu-client

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
  export_to_sponge_config "ng3_commit" "$(git rev-parse HEAD)"
  export_to_sponge_config "ng3_cl_target_branch" "master"
  export_to_sponge_config "ng3_project_id" "pytorchxla"
  export_to_sponge_config "ng3_job_type" "POSTSUBMIT"
  export_to_sponge_config "ng3_test_type" "UNIT"
}

flakiness || true

exit "${test_status}"


