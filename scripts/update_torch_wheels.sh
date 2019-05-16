#!/bin/bash
# Must be YYYYMMDD format
set -e
DATE=$1

DIST_BUCKET="gs://pytorch-tpu-releases/wheels"
TORCH_WHEEL="torch-1.1.0a0+stable-cp35-cp35m-linux_x86_64.whl"
TORCH_XLA_WHEEL="torch_xla-0.1+nightly-cp35-cp35m-linux_x86_64.whl"
STABLE_TORCH_WHEEL="torch-20190513-1.1.0a0+358fb51-cp35-cp35m-linux_x86_64.whl"
STABLE_TORCH_XLA_WHEEL="torch_xla-20190513-0.1+6c24e5b-cp35-cp35m-linux_x86_64.whl"

while getopts 's' OPTION
do
  case $OPTION in
    s)
      echo 'Installing stable build'
      TORCH_WHEEL=$STABLE_TORCH_WHEEL
      TORCH_XLA_WHEEL=$STABLE_TORCH_XLA_WHEEL
      ;;
    *)
      echo 'Usage: ./update_torch_wheel -s (optional)'
      exit 1
      ;;
  esac
done

gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" /tmp/
gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" /tmp/
pip3 install /tmp/"$TORCH_WHEEL"
pip3 install /tmp/"$TORCH_XLA_WHEEL"

rm /tmp/"$TORCH_WHEEL" /tmp/"$TORCH_XLA_WHEEL"
