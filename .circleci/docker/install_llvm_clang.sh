#!/bin/bash

set -e
set -x


function debian_version {
  local VER
  if ! sudo apt-get install -y lsb-release > /dev/null 2>&1 ; then
    VER="buster"
  else
    VER=$(lsb_release -c -s)
  fi
  echo "$VER"
}

function install_llvm_clang() {
  local DEBVER=$(debian_version)

  # Install gcc-11
  # TODO(yeounoh) consolidate different compilers into one, perferably gcc/g++-10,
  # that are also used in ansible. Remove CC/CXX exports afterwards.
  sudo apt-get update
  # Update ppa for GCC
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
  sudo apt update -y
  sudo apt install -y gcc-11 g++-11
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
  sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100
  export CC=gcc-11
  export CXX=g++-11
  export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-11'

  # Update gcov for test coverage
  sudo update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-11 100
  sudo update-alternatives --install /usr/bin/gcov-dump gcov-dump /usr/bin/gcov-dump-11 100
  sudo update-alternatives --install /usr/bin/gcov-tool gcov-tool /usr/bin/gcov-tool-11 100
}

install_llvm_clang
