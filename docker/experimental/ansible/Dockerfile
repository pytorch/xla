ARG python_version=3.8
ARG debian_version=buster

FROM python:${python_version}-${debian_version} AS build

RUN pip install ansible

COPY . /ansible
WORKDIR /ansible

ARG arch=amd64
ARG accelerator=tpu
RUN ansible-playbook playbook.yaml -e "stage=build arch=${arch} accelerator=${accelerator}"

FROM python:${python_version}-${debian_version} AS release

WORKDIR /ansible
COPY . /ansible

ARG arch=amd64
ARG accelerator=tpu
RUN pip install ansible
RUN ansible-playbook playbook.yaml -e "stage=release arch=${arch} accelerator=${accelerator}" --tags "install_deps"

WORKDIR /wheels
COPY --from=build /src/pytorch/dist/*.whl .
COPY --from=build /src/pytorch/xla/dist/*.whl .

RUN pip install *.whl

WORKDIR /
RUN rm -rf /ansible