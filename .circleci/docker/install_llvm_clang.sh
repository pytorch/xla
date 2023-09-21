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
  if ! apt-get install -y -s clang-8 > /dev/null 2>&1 ; then
    maybe_append "deb http://apt.llvm.org/${DEBVER}/ llvm-toolchain-${DEBVER}-8 main" /etc/apt/sources.list
    maybe_append "deb-src http://apt.llvm.org/${DEBVER}/ llvm-toolchain-${DEBVER}-8 main" /etc/apt/sources.list
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
    sudo apt-get update
  fi
  # Build config also sets CC=clang-8, CXX=clang++-8
  sudo apt-get install -y clang-8 clang++-8
  sudo apt-get install -y llvm-8 llvm-8-dev llvm-8-tools

  # TODO(yeounoh) enable clang-10+ for future dev images
  if [[ -e /usr/bin/clang ]]; then
    sudo unlink /usr/bin/clang
  fi
  if [[ -e /usr/bin/clang++ ]]; then
    sudo unlink /usr/bin/clang++
  fi
  sudo ln -s /usr/bin/clang-8 /usr/bin/clang
  sudo ln -s /usr/bin/clang++-8 /usr/bin/clang++
  export CC=clang-8 CXX=clang++-8
}

install_llvm_clang
