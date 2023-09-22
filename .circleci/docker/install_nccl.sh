#!/bin/bash

set -e
set -x

function install_nccl() {
  sudo apt-get install -y --allow-change-held-packages libnccl2 libnccl-dev
}

install_nccl
