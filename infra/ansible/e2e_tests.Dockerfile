ARG python_version=3.8
ARG debian_version=buster

FROM python:${python_version}-${debian_version} AS build

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

ARG ansible_vars

# Build PyTorch and PyTorch/XLA wheels.
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}"

FROM python:${python_version}-${debian_version}
WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

# Install runtime pip and apt dependencies.
RUN ansible-playbook -vvv playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "install_deps"

# Copy test sources.
RUN mkdir -p /src/pytorch/xla
COPY --from=build /src/pytorch/xla/test /src/pytorch/xla/test

# Copy and install wheels.
WORKDIR /tmp/wheels
COPY --from=build /src/pytorch/dist/*.whl ./
COPY --from=build /src/pytorch/xla/dist/*.whl ./

RUN echo "Installing the following wheels" && ls *.whl
RUN pip install *.whl

WORKDIR /

# Clean-up unused directories.
RUN rm -rf /ansible /tmp/wheels