#!/bin/bash

set -ex

export MAX_JOBS=8

PYTORCH_DIR=/tmp/pytorch
XLA_DIR="$PYTORCH_DIR/xla"
git clone --recursive --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
cp -r "$PWD" "$XLA_DIR"

cd $PYTORCH_DIR

# install ninja to speedup the build
pip install ninja

# install Pytorch
patch -p1 < xla/pytorch.patch
python setup.py build develop

# install Bazel which is required by XLA
sudo apt-get -qq update
sudo apt-get -qq install pkg-config zip zlib1g-dev unzip
wget https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh
chmod +x bazel-*.sh
./bazel-*.sh --user
export PATH="$PATH:$HOME/bin"

BAZEL="$(which bazel)"
if [ -z "${BAZEL}" ]; then
  echo "Unable to find bazel..."
  exit 1
fi

# Bazel doesn't work with sccache gcc. https://github.com/bazelbuild/bazel/issues/3642
export CC=/usr/bin/gcc CXX=/usr/bin/g++

# install XLA
pushd "$XLA_DIR"
python setup.py install
popd
