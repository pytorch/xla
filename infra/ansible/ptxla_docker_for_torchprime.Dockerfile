# syntax=docker/dockerfile:1.4
#
# Dockerfile for building a PyTorch/XLA docker image to be used in torchprime
# E2E tests (https://github.com/AI-Hypercomputer/torchprime/actions/workflows/e2e_test.yml)
# triggered from PyTorch/XLA PRs.
#
# This Dockerfile is not used during nightly builds of PyTorch/XLA.
#
# This Dockerfile is also not used by torchprime when a PR is made on torchprime.
# torchprime pins a PyTorch/XLA docker image for use in torchprime PR tests. However,
# when running torchprime tests on PyTorch/XLA PRs, we would override that docker image
# with one built by this Dockerfile.
#
# This Dockerfile is a simplified version of `infra/ansible/Dockerfile`. The latter is meant
# to be run from Cloud Build during nightly triggers. That file would build PyTorch and
# PyTorch/XLA from scratch, and then install those wheels. In contrast, this Dockerfile expects the
# PyTorch and PyTorch/XLA wheels to be built already, which is the case in PR tests.
# The `build-torch-xla` job in the "Build and test" action will have already built the
# wheels, which the `test-torchprime` job will download and make available to this docker build.
#
# The docker image will be pushed to `gcr.io/tpu-pytorch/for-torchprime-ci:${random_id}`. The
# ID is unique for each run of the workflow to avoid interference between concurrent runs.
#
# (Googlers only) Refer to go/ptxla-torchprime-trigger for information on retention policy of
# the docker images.
ARG python_version=3.12
ARG debian_version=bullseye

FROM python:${python_version}-${debian_version}

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

# Build PyTorch and PyTorch/XLA wheels.
ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}"
RUN ansible-playbook -vvv playbook.yaml -e "stage=build_plugin" -e "${ansible_vars}" --skip-tags=fetch_srcs,install_deps

# Install PyTorch wheels. We expect to install three wheels. Example:
# - torch-2.8.0-cp310-cp310-linux_x86_64.whl
# - torch_xla-2.8.0+gitd4b0a48-cp310-cp310-linux_x86_64.whl
# - torchvision-0.22.0a0+966da7e-cp310-cp310-linux_x86_64.whl
# The precise names will depend on the git commit hash used at build time.
WORKDIR /tmp/wheels
COPY ./*.whl ./

RUN echo "Installing the following wheels" && ls *.whl
RUN pip install *.whl

# Install the dependencies including libtpu.
WORKDIR /ansible
RUN pip install ansible
COPY --from=ansible . /ansible

ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}" --tags "install_deps"

WORKDIR /

RUN rm -rf /ansible /tmp/wheels
