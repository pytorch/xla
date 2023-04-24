ARG python_version=3.8
ARG debian_version=buster

FROM python:${python_version}-${debian_version}

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

ARG ansible_vars

# Build and install PyTorch and PyTorch/XLA wheels.
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}"

# Install runtime pip and apt dependencies.
RUN ansible-playbook -vvv playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "install_deps"

WORKDIR /
# Clean-up Ansible setup.
RUN rm -rf /ansible