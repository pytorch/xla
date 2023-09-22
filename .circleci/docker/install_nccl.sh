#!/bin/bash

set -e
set -x

function install_nccl() {
  sudo apt-add-repository ppa:graphics-drivers/ppa
  sudo apt update
  sudo apt install libnccl2=2.18.3-1+cuda12.0.1 libnccl-dev=2.18.3-1+cuda12.0.1_libnccl-2.18.3
  ncclinfo
}

install_nccl
