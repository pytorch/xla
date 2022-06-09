#!/bin/bash

set -ex

source ./xla_env
source .circleci/common.sh

PYTORCH_DIR=/tmp/pytorch
XLA_DIR=$PYTORCH_DIR/xla

# Needs to be kept in sync with .jenkins/pytorch/common_utils.sh in pytorch/pytorch.
TORCHVISION_COMMIT="$(cat $PYTORCH_DIR/.github/ci_commit_pins/vision.txt)"

function pip_install() {
  # retry 3 times
  # old versions of pip don't have the "--progress-bar" flag
  pip install --progress-bar off "$@" || pip install --progress-bar off "$@" || pip install --progress-bar off "$@" ||\
  pip install "$@" || pip install "$@" || pip install "$@"
}

function install_torchvision() {
  pip_install --user "git+https://github.com/pytorch/vision.git@$TORCHVISION_COMMIT"
}

install_torchvision

run_torch_xla_tests $PYTORCH_DIR $XLA_DIR
