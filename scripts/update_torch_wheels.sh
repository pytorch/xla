#!/bin/bash
# Must be YYYYMMDD format
set -e
DATE=$1

DIST_BUCKET="gs://pytorch-tpu-releases/wheels"

TORCH_WHEEL="torch-1.1.0a0+stable-cp35-cp35m-linux_x86_64.whl"
TORCH_XLA_WHEEL="torch_xla-0.1+nightly-cp35-cp35m-linux_x86_64.whl"
if [ ! -z "$DATE" ]; then
  printf 'Which of the following torch wheels would you link to install?\n'
  gsutil ls "gs://pytorch-tpu-releases/wheels/torch-${DATE}-*"
  printf '\n(copy paste and enter one of above): '
  read TORCH_WHEEL
  TORCH_WHEEL="$(basename $TORCH_WHEEL)"

  printf '\nWhich of the following torch_xla wheels would you link to install?\n'
  gsutil ls "gs://pytorch-tpu-releases/wheels/torch_xla-${DATE}-*"
  printf '\n(copy paste and enter one of above): '
  read TORCH_XLA_WHEEL
  TORCH_XLA_WHEEL="$(basename $TORCH_XLA_WHEEL)"
fi

gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" /tmp/
gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" /tmp/
pip3 install /tmp/"$TORCH_WHEEL"
pip3 install /tmp/"$TORCH_XLA_WHEEL"

rm /tmp/"$TORCH_WHEEL" /tmp/"$TORCH_XLA_WHEEL"
