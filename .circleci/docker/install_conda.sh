#!/bin/bash

set -ex

PYTHON_VERSION=$1
CONDA_PREFIX=$2
DEFAULT_PYTHON_VERSION=3.7


function install_and_setup_conda() {
  # Install and configure miniconda
  # Note Miniconda3-lateset is compatible with Python>=3.7
  curl -o miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&  \
      chmod +x miniconda.sh && \
      sudo -H -u jenkins env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
      ./miniconda.sh -b -p "${CONDA_PREFIX}" -u && \
      rm miniconda.sh

  sudo sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
  export PATH="/opt/conda/bin:$PATH"

  echo ". ${CONDA_PREFIX}/etc/profile.d/conda.sh" >> ~/.bashrc
  source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
  ENVNAME="pytorch"
  if conda env list | awk '{print $1}' | grep "^$ENVNAME$"; then
    conda remove --name "$ENVNAME" --all
  fi
  if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION=$DEFAULT_PYTHON_VERSION
  fi
  conda create -y --name "$ENVNAME" python=${PYTHON_VERSION} anaconda
  conda activate "$ENVNAME"
  export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

  conda install -y numpy pyyaml mkl-include setuptools cmake cffi typing \
    tqdm coverage hypothesis dataclasses scipy cython
  /usr/bin/yes | pip install typing_extensions==3.10.0.2  # Required for Python<=3.7
  /usr/bin/yes | pip install --upgrade oauth2client
  /usr/bin/yes | pip install lark-parser
  /usr/bin/yes | pip install --upgrade numpy>=1.14
  /usr/bin/yes | pip install --upgrade numba
  /usr/bin/yes | pip install cloud-tpu-client
  /usr/bin/yes | pip install expecttest==0.1.3
  /usr/bin/yes | pip install ninja  # Install ninja to speedup the build
  /usr/bin/yes | pip install cmake>=3.13 --upgrade  # Using Ninja requires CMake>=3.13
  /usr/bin/yes | pip install absl-py

}

install_and_setup_conda
