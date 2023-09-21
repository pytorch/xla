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

  if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION=$DEFAULT_PYTHON_VERSION
  fi
  export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

  conda update -y -n base conda
  conda install -y python=$PYTHON_VERSION

  conda install -y nomkl numpy pyyaml setuptools cmake git \
    cffi typing tqdm coverage hypothesis dataclasses cython

  /usr/bin/yes | pip install mkl==2022.2.1
  /usr/bin/yes | pip install mkl-include==2022.2.1
  /usr/bin/yes | pip install typing_extensions  # Required for Python<=3.9
  /usr/bin/yes | pip install --upgrade oauth2client
  /usr/bin/yes | pip install lark-parser
  /usr/bin/yes | pip install --upgrade numba
  /usr/bin/yes | pip install cloud-tpu-client
  /usr/bin/yes | pip install expecttest==0.1.3
  /usr/bin/yes | pip install ninja  # Install ninja to speedup the build
  # Using Ninja requires CMake>=3.13, PyTorch requires CMake>=3.18
  /usr/bin/yes | pip install "cmake>=3.18" --upgrade
  /usr/bin/yes | pip install absl-py
  # Additional PyTorch requirements
  /usr/bin/yes | pip install scikit-image scipy
  /usr/bin/yes | pip install boto3
  /usr/bin/yes | pip install mypy
  /usr/bin/yes | pip install psutil
  /usr/bin/yes | pip install unittest-xml-reporting
  /usr/bin/yes | pip install pytest
  /usr/bin/yes | pip install sympy

  sudo apt install -y libtinfo-dev python-is-python3
  conda install -c conda-forge ncurses

  # Ensure /opt/conda/lib/libtinfo.so.6 version information is available.
  sudo apt install -y libtinfo-dev
}

install_and_setup_conda
