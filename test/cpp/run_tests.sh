#!/bin/sh

mkdir build
pushd build
cmake ..
make
./test_ptxla
popd
rm -rf build
