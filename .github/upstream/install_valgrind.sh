#!/bin/bash

set -ex

mkdir valgrind_build && cd valgrind_build
VALGRIND_VERSION=3.16.1
wget https://ossci-linux.s3.amazonaws.com/valgrind-${VALGRIND_VERSION}.tar.bz2
tar -xjf valgrind-${VALGRIND_VERSION}.tar.bz2
cd valgrind-${VALGRIND_VERSION}
./configure --prefix=/usr/local
make -j6
make install
cd ../../
rm -rf valgrind_build
alias valgrind="/usr/local/bin/valgrind"
