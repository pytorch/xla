#!/bin/bash
set -e
set -x

# Activate pytorch-nightly conda env if not already in it
if [ "$CONDA_DEFAULT_ENV" != "pytorch-nightly" ]; then
  conda activate pytorch-nightly
fi

$(dirname $0)/update_torch_wheels.sh
