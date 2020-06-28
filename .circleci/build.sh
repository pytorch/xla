#!/usr/bin/env bash

set -euo pipefail

declare -r REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

source "$REPO_ROOT"'/env'

"$REPO_ROOT"'/.circleci/install_deps.sh'

source "$REPO_ROOT"'/.circleci/pkg_mgr.sh'
source "$REPO_ROOT"'/.circleci/common.sh'

# System default cmake 3.10 cannot find mkl, so point it to the right place.
if command -v conda &>/dev/null; then
  export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
else
  [ -z "${VENV-}" ] && VENV="$HOME"'/.venvs/xla-venv'
  [ ! -d "$VENV" ] && mkdir -p "$VENV" && python3 -m venv "$VENV"
  source "$VENV"'/bin/activate'
  pip install -U pip
  pip install -U setuptools wheel
  pip install mkl
fi

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
if ! git config --global --get user.email &>/dev/null; then
  git config --global user.email "circleci.ossci@gmail.com"
  git config --global user.name "CircleCI"
fi

eval "${SUDO}"' '"${PKG_MGR_EXEC}"' install jq curl'

# Only rebase on runs triggered by PR checks not post-submits.
if [[ -z "${CIRCLE_PROJECT_USERNAME-}" && ! -z "${CIRCLE_PULL_REQUEST-}" ]]; then
  PR_NUM=$(basename $CIRCLE_PULL_REQUEST)
  CIRCLE_PR_BASE_BRANCH=$(curl -s https://api.github.com/repos/$CIRCLE_PROJECT_USERNAME/$CIRCLE_PROJECT_REPONAME/pulls/$PR_NUM | jq -r '.base.ref')
  git rebase 'origin/'"${CIRCLE_PR_BASE_BRANCH}"
  git submodule deinit -f .
  git submodule update --init --recursive
fi

clone_pytorch

cd "$PYTORCH_DIR"
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

# Install hypothesis, required for running some PyTorch test suites
pip install hypothesis

# Install Pytorch without MKLDNN
xla/scripts/apply_patches.sh
python setup.py build develop
sccache --show-stats

# Bazel doesn't work with sccache gcc. https://github.com/bazelbuild/bazel/issues/3642
if ! comamnd -v node; then
  curl -L https://git.io/n-install | bash -s -- -y lts
  export PATH="$PATH:$HOME/n/bin"
  NPM_SUDO=0
  NPM_SUDO_CMD=''
else
  NPM_SUDO=1
  NPM_SUDO_CMD='sudo '
fi

# XLA build requires Bazel
# We use bazelisk to avoid updating Bazel version manually.
eval "${NPM_SUDO_CMD}" npm install -g @bazel/bazelisk
sudo ln -s "$(command -v bazelisk)" /usr/local/bin/bazel

# Install bazels3cache for cloud cache
eval "${NPM_SUDO_CMD}" npm install -g bazels3cache
BAZELS3CACHE="$(which bazels3cache)"
if [ -z "${BAZELS3CACHE}" ]; then
  >&2 echo 'Unable to find bazels3cache...'
  exit 1
fi

>&2 echo 'Installing torchvision at branch master'
rm -rf vision
# TODO: This git clone is bad, it means pushes to torchvision can break
# PyTorch CI
git clone --quiet --depth=10 https://github.com/pytorch/vision
pushd vision
# python setup.py install with a tqdm dependency is broken in the
# Travis Python nightly (but not in latest Python nightlies, so
# this should be a transient requirement...)
# See https://github.com/pytorch/pytorch/issues/7525
#time python setup.py install
pip install -q --user .
popd

bazels3cache --bucket="${XLA_CLANG_CACHE_S3_BUCKET_NAME}" --maxEntrySizeBytes='0' --logging.level='verbose'

# install XLA
pushd "$XLA_DIR"
# Use cloud cache to build when available.
sed -i '/bazel build/ a --remote_http_cache=http://localhost:7777 \\' build_torch_xla_libs.sh

source ./xla_env
python setup.py install
popd
