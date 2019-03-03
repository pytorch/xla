#!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

# Initialize all third_party submodules
git submodule update --init --recursive

# Build torch_xla wheel in conda environment
export NO_CUDA=1 VERSIONED_XLA_BUILD=1
python setup.py bdist_wheel

# Artifacts for pytorch-tpu wheel build collected (as nightly and with date)
ls -lah dist
mkdir build_artifacts
cp dist/* build_artifacts
cd dist && rename "s/\+\w{7}/\+nightly/" *.whl && cd ..
cd build_artifacts && rename "s/^torch_xla/torch_xla-$(date -d "yesterday" +%Y%m%d)/" *.whl && cd ..
mv dist/* build_artifacts
mv build_artifacts/* ../../../

