#!/bin/bash

set -ex

source ./xla_env
source .circleci/common.sh

PYTORCH_DIR=/tmp/pytorch
XLA_DIR=$PYTORCH_DIR/xla

source "$PYTORCH_DIR/.jenkins/pytorch/common_utils.sh"
install_torchvision

run_torch_xla_tests $PYTORCH_DIR $XLA_DIR
