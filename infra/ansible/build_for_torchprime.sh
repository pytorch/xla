#!/bin/bash

set -ex

export IMAGE_NAME="gcr.io/tpu-pytorch/for-torchprime-ci:test"
export DOCKERFILE_PATH="infra/ansible/build_for_torchprime.Dockerfile"
export CONTEXT_PATH="/tmp/wheels"

docker build -t "${IMAGE_NAME}" "${CONTEXT_PATH}" -f "${DOCKERFILE_PATH}" --build-arg ansible_vars='{"arch":"amd64","accelerator":"tpu","pytorch_git_rev":"main","xla_git_rev":"foobar","bundle_libtpu":"0","package_version":"2.8","nightly_release":true}' --build-arg python_version=3.10 --build-arg debian_version=bullseye
docker push "${IMAGE_NAME}"
