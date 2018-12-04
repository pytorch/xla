#!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

# Update compilers to use gcc/g++/cpp-6 as default
sudo apt-get update
sudo apt-get install -y gcc-6
sudo apt-get install -y g++-6
sudo apt-get install -y cpp-6
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 20 --slave /usr/bin/g++ g++ /usr/bin/g++-4.8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10 --slave /usr/bin/g++ g++ /usr/bin/g++-6
sudo update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-6 10
sudo update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-4.8 20
sudo update-alternatives --set gcc /usr/bin/gcc-6
sudo update-alternatives --set cpp /usr/bin/cpp-6
gcc --version
g++ --version
cpp --version

# Remove MPI as this causes problems on other platforms without it
sudo apt-get -y --purge autoremove libboost-mpi-dev
sudo apt-get -y --purge autoremove libboost-mpi-python-dev
sudo apt-get -y --purge autoremove libboost-mpi-python1.54-dev
sudo apt-get -y --purge autoremove libboost-mpi-python1.54.0
sudo apt-get -y --purge autoremove libboost-mpi1.54-dev
sudo apt-get -y --purge autoremove libboost-mpi1.54.0
sudo apt-get -y --purge autoremove libopenmpi-dev
sudo apt-get -y --purge autoremove libopenmpi1.6
sudo apt-get -y --purge autoremove mpi-default-bin
sudo apt-get -y --purge autoremove mpi-default-dev
sudo apt-get -y --purge autoremove openmpi-bin

# Initialize all third_party submodules
git submodule update --init --recursive

# Install required packages for build
sudo apt-get -y install python-pip git
sudo pip install --upgrade google-api-python-client
sudo pip install --upgrade oauth2client

# Install conda environment
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
sh Anaconda3-5.2.0-Linux-x86_64.sh -b
export PATH="$HOME/anaconda3/bin:$PATH"

# Setup conda env
conda create --name pytorch python=3.5 anaconda
source activate pytorch
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing bazel

# Install torch within conda env
# TODO(jysohn): once pytorch/pytorch JIT bug is fixed install nightly wheel instead
sudo /sbin/ldconfig "${HOME}/anaconda3/lib/" "${HOME}/anaconda3/envs/pytorch/lib"
pip install ../../../gfile/torch-1.0.0a0+1ca0ec7-cp35-cp35m-linux_x86_64.whl

# Build pytorch-wheel in conda environment
export NO_CUDA=1
python setup.py bdist_wheel

# Artifacts for pytorch-tpu wheel build collected (as nightly and with date)
ls -lah dist
mkdir build_artifacts
cp dist/* build_artifacts
cd dist && rename "s/\+\w{7}/\+nightly/" *.whl && cd ..
cd build_artifacts && rename "s/^torch_xla/torch_xla-$(date -d "yesterday" +%Y%m%d)/" *.whl && cd ..
mv dist/* build_artifacts
mv build_artifacts/* ../../../

