#!/usr/bin/env bash
set -e
set -x

DIST_BUCKET="gs://tpu-pytorch/wheels"
TORCH_WHEEL="torch-nightly-cp36-cp36m-linux_x86_64.whl"
TORCH_XLA_WHEEL="torch_xla-nightly-cp36-cp36m-linux_x86_64.whl"
TORCHVISION_WHEEL="torchvision-nightly-cp36-cp36m-linux_x86_64.whl"

[[ ! -z "$1" ]] && conda activate $1

function update_wheels() {
  gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" /tmp/
  gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" /tmp/
  gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" /tmp/
  pip install "/tmp/$TORCH_WHEEL"
  pip install "/tmp/$TORCH_XLA_WHEEL"
  pip install "/tmp/$TORCHVISION_WHEEL"

  rm -f "/tmp/$TORCH_WHEEL" "/tmp/$TORCH_XLA_WHEEL" "/tmp/$TORCHVISION_WHEEL"
}

update_wheels
