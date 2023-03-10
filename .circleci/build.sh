#!/bin/bash

set -ex

source ./env
source .circleci/common.sh
PYTORCH_DIR=/tmp/pytorch
XLA_DIR=$PYTORCH_DIR/xla
clone_pytorch $PYTORCH_DIR $XLA_DIR

# Use bazel cache
USE_CACHE=1

SCCACHE="$(which sccache)"
if [ -z "${SCCACHE}" ]; then
  echo "Unable to find sccache..."
  exit 1
fi

if which sccache > /dev/null; then
  # Save sccache logs to file
  sccache --stop-server || true
  rm ~/sccache_error.log || true
  SCCACHE_ERROR_LOG=~/sccache_error.log RUST_LOG=sccache::server=error sccache --start-server

  # Report sccache stats for easier debugging
  sccache --zero-stats
fi

rebase_pull_request_on_target_branch

pushd $PYTORCH_DIR

checkout_torch_pin_if_available

if ! install_deps_pytorch_xla $XLA_DIR $USE_CACHE; then
  exit 1
fi

apply_patches

python -c "import fcntl; fcntl.fcntl(1, fcntl.F_SETFL, 0)"

python setup.py install

sccache --show-stats

source $XLA_DIR/xla_env
export GCLOUD_SERVICE_KEY_FILE="$XLA_DIR/default_credentials.json"
build_torch_xla $XLA_DIR

popd
