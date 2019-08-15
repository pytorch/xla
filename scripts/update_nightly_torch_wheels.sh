#!/bin/bash
set -e
set -x

DIST_BUCKET="gs://tpu-pytorch/wheels"
TORCH_WHEEL="torch-nightly-cp36-cp36m-linux_x86_64.whl"
TORCH_XLA_WHEEL="torch_xla-nightly-cp36-cp36m-linux_x86_64.whl"

# Activate pytorch-nightly conda env if not already in it
if [ "$CONDA_DEFAULT_ENV" != "pytorch-nightly" ]; then
  conda activate pytorch-nightly
fi

function update_wheels() {
  gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" /tmp/
  gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" /tmp/
  pip install "/tmp/$TORCH_WHEEL"
  pip install "/tmp/$TORCH_XLA_WHEEL"

  rm -f "/tmp/$TORCH_WHEEL" "/tmp/$TORCH_XLA_WHEEL"
}

function update_torchvision() {
  # Need to update torchvision too
  # https://github.com/pytorch/vision/issues/967
  torchvision_version=$(pip freeze | grep torchvision | awk -F "==" '{print $2}')
  cd /tmp && \
    git clone -b "v${torchvision_version}" https://github.com/pytorch/vision.git && \
    cd vision && \
    python setup.py install && \
    cd .. && \
    rm -rf /tmp/vision
}

update_wheels
update_torchvision
