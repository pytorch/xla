#!/bin/bash
# Must be YYYYMMDD format
DATE=$1
TORCH_WHEEL="gs://pytorch-tpu-releases/wheels/torch-1.1.0a0+stable-cp35-cp35m-linux_x86_64.whl"
TORCH_XLA_WHEEL="gs://pytorch-tpu-releases/wheels/torch_xla-0.1+nightly-cp35-cp35m-linux_x86_64.whl"
if [ ! -z "$DATE" ]; then
  printf 'Which of the following torch wheels would you link to install?\n'
  gsutil ls "gs://pytorch-tpu-releases/wheels/torch-${DATE}-*"
  printf '\n(copy paste and enter one of above): '
  read TORCH_WHEEL
  printf '\nWhich of the following torch_xla wheels would you link to install?\n'
  gsutil ls "gs://pytorch-tpu-releases/wheels/torch_xla-${DATE}-*"
  printf '\n(copy paste and enter one of above): '
  read TORCH_XLA_WHEEL
fi
gsutil cp "$TORCH_WHEEL" /tmp/.
pip3 install /tmp/"$(basename $TORCH_WHEEL)"
gsutil cp "$TORCH_XLA_WHEEL" /tmp/.
pip3 install /tmp/"$(basename $TORCH_XLA_WHEEL)"
