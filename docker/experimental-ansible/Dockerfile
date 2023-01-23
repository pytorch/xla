ARG python_version=3.8
ARG debian_version=buster

FROM python:${python_version}-${debian_version} AS build

RUN pip install ansible

COPY ansible /ansible
WORKDIR /ansible

RUN ansible-playbook playbook.yaml -e "stage=build arch=amd64 accelerator=tpu"

FROM build AS release

WORKDIR /ansible
RUN ansible-playbook playbook.yaml -e "stage=release arch=amd64 accelerator=tpu"
