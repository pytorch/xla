#!/bin/bash

set -e
set -x

LLVM_RAW_ARCHIVE=$1


# Local llvm cache in case the tensorflow mirror becomes unavailable
function download_llvm_raw_archive() {
  # Extract the xla pinned tensorflow/third_party/llvm version,
  # avoid doing this manually.
  git clone --recursive --quiet https://github.com/pytorch/xla.git
  local LLVM_COMMIT=$(cat xla/third_party/tensorflow/third_party/llvm/workspace.bzl \
    | grep "LLVM_COMMIT =" | awk '{print $3}' | sed 's/"//g')
  curl --fail "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/${LLVM_COMMIT}.tar.gz" \
    --output "${LLVM_RAW_ARCHIVE}/${LLVM_COMMIT}.tar.gz"
  rm -rf xla
}

download_llvm_raw_archive
