#!/bin/bash

set -ex

source ./xla_env
source .circleci/common.sh

PYTORCH_DIR="${KOKORO_ARTIFACTS_DIR}/github/pytorch"
XLA_DIR=$PYTORCH_DIR/xla
source ${XLA_DIR}/.kokoro/common.sh
TORCHVISION_COMMIT="$(cat $PYTORCH_DIR/.github/ci_commit_pins/vision.txt)"

function pip_install() {
  # retry 3 times
  # old versions of pip don't have the "--progress-bar" flag
  pip install --progress-bar off "$@" || pip install --progress-bar off "$@" || pip install --progress-bar off "$@" ||\
  pip install "$@" || pip install "$@" || pip install "$@"
}

function install_torchvision() {
  pip_install --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@$TORCHVISION_COMMIT"
}

install_torchvision

run_torch_xla_tests $PYTORCH_DIR $XLA_DIR
