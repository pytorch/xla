#!/bin/sh

export TF_CPP_MIN_LOG_LEVEL=1

mkdir build
pushd build
cmake ..
make
./test_ptxla
popd
rm -rf build
