# Dockerfile for building a development image.
# The built image contains all required pip and apt packages for building and
# running PyTorch and PyTorch/XLA. The image doesn't contain any source code.
ARG python_version=3.8
ARG debian_version=bookworm

FROM python:${python_version}-${debian_version}

RUN pip install ansible

COPY . /ansible
WORKDIR /ansible

# Ansible 2.19 requires this environment variable being set, so that we can use
# string variables as boolean. 
ENV ALLOW_BROKEN_CONDITIONALS "1"
# List Ansible tasks to apply for the dev image.
ENV TAGS="bazel,configure_env,install_deps"

ARG ansible_vars
RUN ansible-playbook playbook.yaml -e "stage=build" -e "${ansible_vars}" --tags "${TAGS}"
RUN ansible-playbook playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "${TAGS}"
