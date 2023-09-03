#!/bin/bash

function run_deployment_tests() {
  export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  export XRT_WORKERS="localservice:0;grpc://localhost:40934"
  export CC=clang-9 CXX=clang++-9

  # We don't need to load libtpu since test is being done on CPU.
  time TPU_LOAD_LIBRARY=0 python /pytorch/xla/test/test_train_mp_mnist.py --fake_data
  # time TPU_LOAD_LIBRARY=0 bash /pytorch/xla/test/run_tests.sh
  RUN_TEST_CMD=$(find /pytorch -name run_tests.sh | grep xla/test/cpp/ | tail -1)
  # TODO: C++ tests are disabled since some (e.g., TestSymEig) tests are failing
  # on the cloud enviorment.
  # time bash ${RUN_TEST_CMD}
}

function collect_wheels() {
  sudo apt-get install -y rename

  release_version=$1
  wheel_version="${release_version}"
  if [ "${release_version}" != "nightly" ]; then
    wheel_version=$( echo "${release_version}" | grep -oP '\d+.\d+(.\d+)?' )
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
  # pushd /tmp/staging-wheels
  # cp /pytorch/audio/dist/*.whl .
  # rename -v "s/^torchaudio-(.*?)-cp/torchaudio-${wheel_version}-cp/" *.whl
  # popd
  # mv /tmp/staging-wheels/* .

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
  # pushd /pytorch/audio/dist
  # rename -v "s/^torchaudio-(.*?)-cp/torchaudio-${wheel_version}+$(date -u +%Y%m%d)-cp/" *.whl
  # popd

  cp /pytorch/dist/*.whl ./ && cp /pytorch/xla/dist/*.whl ./ && cp /pytorch/vision/dist/*.whl ./
  # && cp /pytorch/audio/dist/*.whl ./
}
