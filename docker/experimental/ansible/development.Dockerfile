# Dockerfile for building a development image.
# The built image contains all required pip and apt packages for building and
# running PyTorch and PyTorch/XLA. The image doesn't contain any source code.
ARG python_version=3.8
ARG debian_version=buster

FROM python:${python_version}-${debian_version}

RUN pip install ansible

COPY . /ansible
WORKDIR /ansible

ARG arch=amd64
ARG accelerator=tpu

RUN ansible-playbook playbook.yaml -e "stage=build arch=${arch} accelerator=${accelerator}" --skip-tags "fetch_srcs,build_srcs"
RUN ansible-playbook playbook.yaml -e "stage=release arch=${arch} accelerator=${accelerator}" --skip-tags "fetch_srcs,build_srcs"