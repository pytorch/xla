# Dockerfile for building a development image.
# The built image contains all required pip and apt packages for building and
# running PyTorch and PyTorch/XLA. The image doesn't contain any source code.
ARG python_version=3.8
ARG debian_version=buster
ARG ansible_vars

FROM python:${python_version}-${debian_version}

RUN pip install ansible

COPY . /ansible
WORKDIR /ansible

ARG arch=amd64
ARG accelerator=tpu

# List Asnible tasks to apply for the dev image.
ENV TAGS="bazel,configure_env,install_deps"

RUN ansible-playbook playbook.yaml -e "stage=build" -e "${ansible_vars}" --tags "${TAGS}"
RUN ansible-playbook playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "${TAGS}"
