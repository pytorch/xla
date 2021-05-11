#!/bin/bash

set -ex

source ./env
source .circleci/common.sh
clone_pytorch
source "$PYTORCH_DIR/.jenkins/pytorch/common_utils.sh"

# System default cmake 3.10 cannot find mkl, so point it to the right place.
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

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

# TODO: directly use ENV_VAR when CircleCi exposes base branch.
# Try rebasing on top of base (dest) branch first.
# This allows us to pickup the latest fix for PT-XLA breakage.
# Also it might improve build time as we have warm cache.
git config --global user.email "circleci.ossci@gmail.com"
git config --global user.name "CircleCI"
sudo apt-get update && sudo apt-get -qq install jq
# Only rebase on runs triggered by PR checks not post-submits.
if [[ ! -z "${CIRCLE_PULL_REQUEST}" ]]; then
  PR_NUM=$(basename $CIRCLE_PULL_REQUEST)
  CIRCLE_PR_BASE_BRANCH=$(curl -s https://api.github.com/repos/$CIRCLE_PROJECT_USERNAME/$CIRCLE_PROJECT_REPONAME/pulls/$PR_NUM | jq -r '.base.ref')
  git rebase "origin/${CIRCLE_PR_BASE_BRANCH}"
  git submodule deinit -f .
  git submodule update --init --recursive
fi

cd $PYTORCH_DIR
# Checkout specific commit ID/branch if pinned.
COMMITID_FILE="xla/.torch_pin"
if [ -e "$COMMITID_FILE" ]; then
  git checkout $(cat "$COMMITID_FILE")
fi
git submodule update --init --recursive

# Install ninja to speedup the build
pip install ninja

# Install the Lark parser required for the XLA->ATEN Type code generation.
pip install lark-parser

# Install libraries required for running some PyTorch test suites
pip install hypothesis
pip install cloud-tpu-client

# Install Pytorch without MKLDNN
xla/scripts/apply_patches.sh
python setup.py build install
sccache --show-stats

# Bazel doesn't work with sccache gcc. https://github.com/bazelbuild/bazel/issues/3642
sudo apt-get -qq update

sudo apt-get -qq install npm nodejs

# XLA build requires Bazel
# We use bazelisk to avoid updating Bazel version manually.
sudo npm install -g @bazel/bazelisk
sudo ln -s "$(command -v bazelisk)" /usr/bin/bazel

# Install bazels3cache for cloud cache
sudo npm install -g bazels3cache
BAZELS3CACHE="$(which bazels3cache)"
if [ -z "${BAZELS3CACHE}" ]; then
  echo "Unable to find bazels3cache..."
  exit 1
fi

echo "Installing torchvision at branch master"
install_torchvision

bazels3cache --bucket=${XLA_CLANG_CACHE_S3_BUCKET_NAME} --maxEntrySizeBytes=0 --logging.level=verbose

# install XLA
pushd "$XLA_DIR"
# Use cloud cache to build when available.
sed -i '/bazel build/ a --remote_http_cache=http://localhost:7777 \\' build_torch_xla_libs.sh

source ./xla_env
python setup.py install
popd
