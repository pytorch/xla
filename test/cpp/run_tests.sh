#!/bin/sh

LOGFILE=/tmp/pytorch_cpp_test.log

mkdir build
pushd build
cmake ..
make
./test_ptxla 2>$LOGFILE
popd
rm -rf build
