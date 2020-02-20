#!/bin/bash
set -e
set -x

# Activate torch-xla-nightly conda env if not already in it
if [ "$CONDA_DEFAULT_ENV" != "torch-xla-nightly" ]; then
  conda activate torch-xla-nightly
fi

$(dirname ${BASH_SOURCE[0]})/update_torch_wheels.sh
