#!/bin/bash

set -ex

# Check required environment variables
if [ -z "${DEFAULT_CONTEXT_PATH}" ]; then
  echo "ERROR: DEFAULT_CONTEXT_PATH is not set"
  exit 1
fi
if [ -z "${DOCKER_IMAGE_NAME}" ]; then
  echo "ERROR: DOCKER_IMAGE_NAME is not set"
  exit 1
fi
if [ -z "${DOCKER_IMAGE_TAG}" ]; then
  echo "ERROR: DOCKER_IMAGE_TAG is not set"
  exit 1
fi
if [ -z "${DOCKER_PROJECT}" ]; then
  echo "ERROR: DOCKER_PROJECT is not set"
  exit 1
fi

export IMAGE_NAME="gcr.io/${DOCKER_PROJECT}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
export DOCKERFILE_PATH="infra/ansible/build_for_torchprime.Dockerfile"

echo "Building and pushing image: ${IMAGE_NAME}"

# Define ansible vars used in the docker file
# TODO(yifeit): read versions from git
read -r -d '' ANSIBLE_VARS_JSON << EOM || { exit_code=$?; [[ $exit_code -eq 1 ]]; }
{
  "arch": "amd64",
  "accelerator": "tpu",
  "pytorch_git_rev": "main",
  "xla_git_rev": "foobar",
  "bundle_libtpu": "0",
  "package_version": "2.8",
  "nightly_release": true
}
EOM
ANSIBLE_VARS_COMPACT=$(echo "$ANSIBLE_VARS_JSON" | tr -d '\n' | tr -d ' ')

docker build -t "${IMAGE_NAME}" \
    --build-context ansible=infra/ansible \
    "${DEFAULT_CONTEXT_PATH}" \
    -f "${DOCKERFILE_PATH}" \
    --build-arg ansible_vars="${ANSIBLE_VARS_COMPACT}" \
    --build-arg python_version=3.10 \
    --build-arg debian_version=bullseye
docker push "${IMAGE_NAME}"

echo "Successfully pushed image: ${IMAGE_NAME}"
