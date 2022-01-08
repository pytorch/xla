#!/bin/bash

set -e
set -x

PYTHON_VERSION=$1
CONDA_PREFIX=$2
DEFAULT_PYTHON_VERSION=3.6


function install_and_setup_conda() {
  # Install conda if dne already.
  CONDA_VERSION="5.3.1"
  curl -O "https://repo.anaconda.com/archive/Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"
  sh "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh" -u -b -p "${CONDA_PREFIX}"
  rm -f "Anaconda3-${CONDA_VERSION}-Linux-x86_64.sh"

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
    typing_extensions tqdm coverage hypothesis dataclasses
  /usr/bin/yes | pip install typing_extensions==3.10.0.2
  /usr/bin/yes | pip install --upgrade oauth2client
  /usr/bin/yes | pip install lark-parser
  /usr/bin/yes | pip install cloud-tpu-client
  /usr/bin/yes | pip install expecttest==0.1.3
}

install_and_setup_conda
