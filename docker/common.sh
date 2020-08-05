#!/bin/bash

function run_deployment_tests() {
  export CPU_NUM_DEVICES=4
  export CC=clang-8 CXX=clang++-8

  time python /pytorch/xla/test/test_train_mp_mnist.py
  time bash /pytorch/xla/test/run_tests.sh
  time bash /pytorch/xla/test/cpp/run_tests.sh
}

function collect_wheels() {
  sudo apt-get install -y rename

  release_version=$1
  wheel_version="${release_version}"
  if [ "${release_version}" != "nightly" ]; then
    wheel_version=$( echo "${release_version}" | grep -oP '\d+.\d+' )
  fi

  mkdir /tmp/staging-wheels

  pushd /tmp/staging-wheels
  cp /pytorch/dist/*.whl .
  rename -v "s/^torch-(.*?)-cp/torch-${wheel_version}-cp/" *.whl
  popd
  mv /tmp/staging-wheels/* .
  pushd /tmp/staging-wheels
  cp /pytorch/xla/dist/*.whl .
  rename -v "s/^torch_xla-(.*?)-cp/torch_xla-${wheel_version}-cp/" *.whl
  popd
  mv /tmp/staging-wheels/* .
  pushd /tmp/staging-wheels
  cp /pytorch/vision/dist/*.whl .
  rename -v "s/^torchvision-(.*?)-cp/torchvision-${wheel_version}-cp/" *.whl
  popd
  mv /tmp/staging-wheels/* .

  rm -rf /tmp/staging-wheels

  pushd /pytorch/dist
  rename -v "s/^torch-(.*?)-cp/torch-${wheel_version}+$(date -u +%Y%m%d)-cp/" *.whl
  popd
  pushd /pytorch/xla/dist
  rename -v "s/^torch_xla-(.*?)-cp/torch_xla-${wheel_version}+$(date -u +%Y%m%d)-cp/" *.whl
  popd
  pushd /pytorch/vision/dist
  rename -v "s/^torchvision-(.*?)-cp/torchvision-${wheel_version}+$(date -u +%Y%m%d)-cp/" *.whl
  popd
  cp /pytorch/dist/*.whl ./ && cp /pytorch/xla/dist/*.whl ./ && cp /pytorch/vision/dist/*.whl ./
}
