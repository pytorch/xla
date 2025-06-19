#!/bin/bash

# This script rebuilds torch_xla (including torchax). It can also rebuild
# pytorch and torchvision if requested. To see the usage, run it with the
# -h flag.

set -e # Fail on any error.

# Absolute path to the directory of this script.
_SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Which project to start the build from. We assume that the project
# dependencies are: pytorch <= vision <= torch_xla. By default,
# we assume that the user is working on torch_xla and there's
# no need to rebuild pytorch or vision. However, the user can override
# the "base" project using the -b flag.
_BUILD_BASE=torch_xla

# Parse commandline flags.
while getopts 'ab:ht' OPTION; do
  case $OPTION in
  a)
    _BUILD_BASE="pytorch"
    ;;
  b)
    _BUILD_BASE=$OPTARG
    # Validate _BUILD_BASE.
    case "$_BUILD_BASE" in
    "pytorch" | "vision" | "torch_xla") ;;
    *)
      echo "ERROR: Invalid <base_project>: $_BUILD_BASE. Must be one of pytorch, vision, or torch_xla."
      exit 1
      ;;
    esac
    ;;
  h)
    echo "Usage: $0 [-b <base_project>] [-a] [-t]"
    echo "where"
    echo "  -b <base_project> selects which project to start the build from. <base_project> be pytorch, vision, or torch_xla (the default)."
    echo "  -a (short for -b pytorch) builds all projects."
    echo "  -t checks that the torch_xla and torchax libraries are built and installed successfully."
    exit 0
    ;;
  t)
    _TEST_INSTALL=1
    ;;
  \?) # This catches all invalid options.
    exit 1 ;;
  esac
done

set -x # Display commands being run.

# Rebuild pytorch if _BUILD_BASE is pytorch.
if [ "$_BUILD_BASE" == "pytorch" ]; then
  # Change to the pytorch directory.
  cd $_SCRIPT_DIR/../..

  # Remove any leftover old wheels and old installation.
  pip uninstall torch -y
  python3 setup.py clean

  # Install pytorch
  python3 setup.py bdist_wheel
  python3 setup.py install
fi

# Rebuild torchvision if _BUILD_BASE is pytorch or vision.
if [ "$_BUILD_BASE" == "pytorch" ] || [ "$_BUILD_BASE" == "vision" ]; then
  _VISION_DIR=$_SCRIPT_DIR/../../../vision
  if [ -d $_VISION_DIR ]; then
    cd $_VISION_DIR
    python3 setup.py develop
  fi
fi

# Rebuild torch_xla.

cd $_SCRIPT_DIR/..
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

if [ "$_TEST_INSTALL" == "1" ]; then
  # Test that the libraries are installed correctly.
  python3 -c 'import torch_xla; print(torch_xla.devices()); import torchax; torchax.enable_globally()'
fi
