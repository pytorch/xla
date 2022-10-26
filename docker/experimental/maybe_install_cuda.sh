#!/bin/bash
set -x
set -e

cuda_version_hyphenated=${CUDA_VERSION//./-}
cudnn_major_version=${CUDNN_VERSION%%.*}

function common() {
  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/3bf863cc.pub
  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/ /" >> /etc/apt/sources.list
  apt-get update

  apt-get install -y cuda-libraries-${cuda_version_hyphenated} cuda-minimal-build-${cuda_version_hyphenated} libcudnn${cudnn_major_version}=${CUDNN_VERSION}-1+cuda${CUDA_VERSION}
}

function build() {
  common

  apt-get install -y libcudnn${cudnn_major_version}-dev=${CUDNN_VERSION}-1+cuda${CUDA_VERSION} cuda-toolkit-${cuda_version_hyphenated}
}

if [ "$CUDA" ==  "1" ]; then
  set -u
  ${@:1}
else
  echo "Skipping CUDA installation (\$CUDA=$CUDA)"
fi
