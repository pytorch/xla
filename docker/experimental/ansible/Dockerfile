ARG python_version=3.8
ARG debian_version=buster

FROM python:${python_version}-${debian_version} AS build

RUN pip install ansible

COPY . /ansible
WORKDIR /ansible

ARG arch=amd64
ARG accelerator=tpu
ARG cuda_version=11.8
ARG pytorch_git_rev=HEAD
ARG xla_git_rev=HEAD
ARG nightly_release=false
ARG package_version

RUN ansible-playbook -vvv playbook.yaml -e \
  "stage=build \
  arch=${arch} \
  accelerator=${accelerator} \
  cuda_version=${cuda_version} \
  pytorch_git_rev=${pytorch_git_rev} \
  xla_git_rev=${xla_git_rev} \
  nightly_release=${nightly_release} \
  package_version=${package_version}"

FROM python:${python_version}-${debian_version} AS release

WORKDIR /ansible
COPY . /ansible

ARG arch=amd64
ARG accelerator=tpu
ARG cuda_version=11.8
ARG pytorch_git_rev=HEAD
ARG xla_git_rev=HEAD

RUN pip install ansible
RUN ansible-playbook -vvv playbook.yaml -e \
  "stage=release \
  arch=${arch} \
  accelerator=${accelerator} \
  cuda_version=${cuda_version} \
  pytorch_git_rev=${pytorch_git_rev} \
  xla_git_rev=${xla_git_rev} \
  " --tags "install_deps"

WORKDIR /dist
COPY --from=build /dist/*.whl /dist/

RUN echo "Installing the following wheels" && ls /dist/*.whl
RUN pip install *.whl

WORKDIR /
RUN rm -rf /ansible
