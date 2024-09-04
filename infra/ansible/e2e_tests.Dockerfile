ARG python_version=3.8
ARG debian_version=bullseye

FROM python:${python_version}-${debian_version} AS build

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

# Build PyTorch and PyTorch/XLA wheels.
ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}"
RUN ansible-playbook -vvv playbook.yaml -e "stage=build_plugin" -e "${ansible_vars}" --skip-tags=fetch_srcs,install_deps

FROM python:${python_version}-${debian_version}
WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

# Copy test sources.
RUN mkdir -p /src/pytorch/xla
COPY --from=build /src/pytorch/xla/test /src/pytorch/xla/test
COPY --from=build /src/pytorch/xla/examples /src/pytorch/xla/examples

# Copy and install wheels.
WORKDIR /tmp/wheels
COPY --from=build /dist/*.whl ./

RUN echo "Installing the following wheels" && ls *.whl
RUN pip install *.whl

# Install runtime pip and apt dependencies.
ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "install_deps"

WORKDIR /

# Clean-up unused directories.
RUN rm -rf /ansible /tmp/wheels
