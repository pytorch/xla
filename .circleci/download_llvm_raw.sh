#!/bin/bash

set -e
set -x

# Local llvm cache in case the tensorflow mirror becomes unavailable
function download_llvm_raw_archive() {
  # Extract the xla pinned tensorflow/third_party/llvm version,
  # avoid doing this manually.
  git clone --recursive --quiet https://github.com/pytorch/xla.git
  local LLVM_COMMIT=$(cat xla/third_party/tensorflow/third_party/llvm/workspace.bzl \
    | grep "LLVM_COMMIT =" | awk '{print $3}' | sed 's/"//g')

  if [ "$(gsutil -q stat gs://tpu-pytorch/llvm-raw/${LLVM_COMMIT}.tar.gz; echo $?)" -eq "1" ]; then
    http_code="$(curl --fail -w '%{http_code}\n' \
      https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/${LLVM_COMMIT}.tar.gz \
      --output ${LLVM_COMMIT}.tar.gz)"
    if [ "${http_code}" -eq "404" ]; then
      echo "mirror.tensorflow.org/github.com/llvm/ is not available."
      echo "If the issue persists, plesase report at github.com/pytorch/xla"
      exit 1
    fi
    gsutil cp "${LLVM_COMMIT}.tar.gz" "gs://tpu-pytorch/llvm-raw/${LLVM_COMMIT}.tar.gz"
  fi
  rm -rf xla
}

download_llvm_raw_archive
