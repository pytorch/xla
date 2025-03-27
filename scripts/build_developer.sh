#!/bin/bash

set -e  # Fail on any error.
set -x  # Display commands being run.

# Change to pytorch directory.
cd "$(dirname "$(readlink -f "$0")")"
cd ../../

# First remove any left over old wheels
# and old installation
pip uninstall torch -y
python3 setup.py clean

# Install pytorch
python3 setup.py bdist_wheel
python3 setup.py install
cd ..

# Optionally install torchvision.
if [ -d "vision" ]; then
  cd vision
  python3 setup.py develop
  cd .. 
fi

# Install torch_xla
cd pytorch/xla
pip uninstall torch_xla torchax torch_xla2 -y

# Build the wheel too, which is useful for other testing purposes.
python3 setup.py bdist_wheel

# Link the source files for local development.
python3 setup.py develop

# libtpu is needed to talk to the TPUs. If TPUs are not present,
# installing this wouldn't hurt either.
pip install torch_xla[tpu] \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://storage.googleapis.com/libtpu-releases/index.html

# Install Pallas dependencies
pip install torch_xla[pallas] \
  -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
  -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

# Test that the library is installed correctly.
python3 -c 'import torch_xla; import torchax; print(torch_xla.device())'
