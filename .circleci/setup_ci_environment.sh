#!/bin/bash

set -e
set -x

pip install --upgrade pip
pip install pyyaml -qqq
echo "export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_FOR_ECR_READ_WRITE}" >> $BASH_ENV
echo "export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY_FOR_ECR_READ_WRITE}" >> $BASH_ENV
echo "export WORKDIR=/var/lib/jenkins/workspace" >> $BASH_ENV


# Install google-cloud-sdk
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get -y install google-cloud-sdk
echo $GCLOUD_SERVICE_KEY | gcloud auth activate-service-account --key-file=-
/usr/bin/yes | gcloud auth configure-docker

# Set up Docker repo
sudo apt-get install lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Set up NVIDIA docker repo
curl -s -L --retry 3 https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
echo "deb https://nvidia.github.io/nvidia-container-runtime/ubuntu20.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/libnvidia-container/ubuntu20.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/nvidia-docker/ubuntu20.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list

# Remove unnecessary sources
sudo rm -f /etc/apt/sources.list.d/google-chrome.list
sudo rm -f /etc/apt/heroku.list
sudo rm -f /etc/apt/openjdk-r-ubuntu-ppa-xenial.list
sudo rm -f /etc/apt/partner.list

retry () {
    $*  || $* || $* || $* || $*
}

# Method adapted from here: https://askubuntu.com/questions/875213/apt-get-to-retry-downloading
# (with use of tee to avoid permissions problems)
# This is better than retrying the whole apt-get command
echo "APT::Acquire::Retries \"3\";" | sudo tee /etc/apt/apt.conf.d/80-retries

sudo apt-get -y update
sudo apt-get -y remove linux-image-generic linux-headers-generic linux-generic docker-ce
# WARNING: Docker version is hardcoded here; you must update the
# version number below for docker-ce and nvidia-docker2 to get newer
# versions of Docker.  We hardcode these numbers because we kept
# getting broken CI when Docker would update their docker version,
# and nvidia-docker2 would be out of date for a day until they
# released a newer version of their package.
#
# How to figure out what the correct versions of these packages are?
# My preferred method is to start a Docker instance of the correct
# Ubuntu version (e.g., docker run -it ubuntu:16.04) and then ask
# apt what the packages you need are.  Note that the CircleCI image
# comes with Docker.
#
# Using 'retry' here as belt-and-suspenders even though we are
# presumably retrying at the single-package level via the
# apt.conf.d/80-retries technique.
retry sudo apt-get -y install \
  linux-headers-$(uname -r) \
  linux-image-generic \
  moreutils \
  docker-ce=5:20.10.13~3-0~ubuntu-focal \
  nvidia-container-runtime=3.8.1-1 \
  nvidia-docker2=2.9.1-1 \
  expect-dev

sudo pkill -SIGHUP dockerd


retry sudo pip -q install awscli==1.16.35

if [ -n "${USE_CUDA_DOCKER_RUNTIME:-}" ]; then
  DRIVER_FN="NVIDIA-Linux-x86_64-450.80.02.run"
  wget "https://download.nvidia.com/XFree86/Linux-x86_64/450.80.02/$DRIVER_FN"
  sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
  nvidia-smi
fi
