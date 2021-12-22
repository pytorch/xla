#!/bin/bash

set -ex

source ./xla_env
source .circleci/common.sh

PYTORCH_DIR=/tmp/pytorch
XLA_DIR=$PYTORCH_DIR/xla

sudo apt-get -qq update
sudo apt-get -qq clang-9
sudo apt-get -qq clang++-9

source "$PYTORCH_DIR/.jenkins/pytorch/common_utils.sh"
install_torchvision

run_torch_xla_tests $PYTORCH_DIR $XLA_DIR
