ARG python_version=3.8
ARG debian_version=buster

FROM python:${python_version}-${debian_version} AS build

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}"
RUN ansible-playbook -vvv playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "install_deps"
