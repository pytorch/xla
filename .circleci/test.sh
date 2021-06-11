#!/bin/bash

set -ex

source ./xla_env
source .circleci/common.sh

PYTORCH_DIR=/tmp/pytorch
XLA_DIR=$PYTORCH_DIR/xla
run_torch_xla_tests $PYTORCH_DIR $XLA_DIR
