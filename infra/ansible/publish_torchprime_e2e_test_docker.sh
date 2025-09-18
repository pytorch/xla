#!/bin/bash

# This script builds and pushes a docker image to be used for torchprime E2E tests.
#
# torchprime is a reference implementation of models using PyTorch/XLA:
# https://github.com/AI-Hypercomputer/torchprime.
#
# The purpose of building a docker image here is to trigger torchprime E2E tests
# from PyTorch/XLA PRs and post-submits. The reason for running torchprime tests
# on PyTorch/XLA changes is to ensure that torchprime models are not broken.
# See https://github.com/AI-Hypercomputer/torchprime/issues/161 for the detailed
# motivation.
#
# The docker image will be pushed to
# `gcr.io/${DOCKER_PROJECT}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}`. By default, the
# `torchprime-e2e-test` job in the `.github/workflows/_torchprime_ci.yml` workflow will
# configure the env vars such that the image is pushed to
# `gcr.io/tpu-pytorch/for-torchprime-ci:${random_id}`. The ID is unique for each run
# of the workflow to avoid interference between concurrent runs.
#
# (Googlers only) Refer to go/ptxla-torchprime-trigger for information on retention policy of
# the docker images.

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

export DOCKER_URL="gcr.io/${DOCKER_PROJECT}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
export DOCKERFILE_PATH="infra/ansible/ptxla_docker_for_torchprime.Dockerfile"

echo "Building and pushing image: ${DOCKER_URL}"

# Define ansible vars used in the docker file by `ansible-playbook`.
#
# See `infra/ansible/playbook.yaml` and `infra/ansible/config/vars.yaml`
# for definition of the variables.
read -r -d '' ANSIBLE_VARS_JSON << EOM || { exit_code=$?; [[ $exit_code -eq 1 ]]; }
{
  "arch": "amd64",
  "accelerator": "tpu",
  "bundle_libtpu": "0",
  "git_versioned_xla_build": true,
  "nightly_release": true
}
EOM
ANSIBLE_VARS_COMPACT=$(echo "$ANSIBLE_VARS_JSON" | tr -d '\n' | tr -d ' ')

docker build -t "${DOCKER_URL}" \
    --build-context ansible=infra/ansible \
    "${DEFAULT_CONTEXT_PATH}" \
    -f "${DOCKERFILE_PATH}" \
    --build-arg ansible_vars="${ANSIBLE_VARS_COMPACT}" \
    --build-arg python_version=3.12 \
    --build-arg debian_version=bullseye
docker push "${DOCKER_URL}"

echo "Successfully pushed image: ${DOCKER_URL}"
