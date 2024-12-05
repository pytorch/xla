#!/bin/bash

set -e  # Fail on any error.
set -x  # Display commands being run.

# Change to pytorch directory.
cd "$(dirname "$(readlink -f "$0")")"
cd ../../

python3 setup.py bdist_wheel
python3 setup.py install
cd ..

# Optionally install torchvision.
if [ -d "vision" ]; then
  cd vision
  python3 setup.py develop
fi

cd ..
cd pytorch/xla
python3 setup.py develop

# libtpu is needed to talk to the TPUs. If TPUs are not present,
# installing this wouldn't hurt either.
pip install torch_xla[tpu] \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://storage.googleapis.com/libtpu-releases/index.html

# Test that the library is installed correctly.
python3 -c 'import torch_xla as xla; print(xla.device())'

