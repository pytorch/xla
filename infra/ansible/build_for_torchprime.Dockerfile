# syntax=docker/dockerfile:1.4
ARG python_version=3.10
ARG debian_version=bullseye

FROM python:${python_version}-${debian_version} AS release

WORKDIR /tmp/wheels
COPY ./*.whl ./

RUN echo "Installing the following wheels" && ls *.whl
RUN pip install *.whl

# Install the dependencies including libtpu.
WORKDIR /ansible
RUN pip install ansible
COPY --from=ansible . /ansible

ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "install_deps"

WORKDIR /

RUN rm -rf /ansible /tmp/wheels
