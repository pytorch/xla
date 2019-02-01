!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

# Setup build environment
## Update compilers to use gcc/g++/cpp-6 as default
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

## Remove MPI as this causes problems on other platforms without it
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

## Install required packages for build
sudo apt-get -y install python-pip git
sudo pip install --upgrade google-api-python-client
sudo pip install --upgrade oauth2client

## Install conda environment
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
sh Anaconda3-5.2.0-Linux-x86_64.sh -b
export PATH="$HOME/anaconda3/bin:$PATH"

## Setup conda env
conda create --name pytorch python=3.5 anaconda
source activate pytorch
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing bazel
sudo /sbin/ldconfig "${HOME}/anaconda3/lib/" "${HOME}/anaconda3/envs/pytorch/lib"

# Install the Lark parser required for the XLA->ATEN Type code generation.
pip install lark-parser

# Place pytorch/xla under pytorch/pytorch
cd github
git clone --recursive https://github.com/pytorch/pytorch.git
mv xla/ pytorch/
cd pytorch

# TODO(jysohn): remove following patching once pytorch JIT bug is fixed
git checkout $(cat xla/.torch_commit_id)
git apply xla/pytorch.patch
# Build and install torch wheel and collect artifact
export NO_CUDA=1
python setup.py bdist_wheel
pip install dist/*.whl
#pip install dist/torch-1.1.0a0+$(head -c 7 xla/.torch_commit_id)-cp35-cp35m-linux_x86_64.whl
cd dist && rename "s/\+\w{7}/\+stable/" *.whl && cd ..
mv dist/* ../../

# Execute torch_xla build
cd xla
./kokoro_build.sh
