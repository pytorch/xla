ARG python_version=3.8
ARG debian_version=bullseye

FROM python:${python_version}-${debian_version} AS build

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

ARG ansible_vars

# Add NVIDIA repository and update
RUN apt-get update && \
    apt-get install -y gnupg curl ca-certificates && \
    apt-get clean && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring.deb && \
    dpkg -i cuda-keyring.deb && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update

RUN apt list -a libcudnn8 && \
    apt list -a libcudnn9-cuda-12

# HACK: install build dependencies only, but skip build step
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}" --tags "bazel,configure_env,install_deps"
RUN ansible-playbook -vvv playbook.yaml -e "stage=build_plugin" -e "${ansible_vars}" 
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}" --skip-tags=fetch_srcs,install_deps

FROM python:${python_version}-${debian_version} AS release

WORKDIR /tmp/wheels
COPY --from=build /src/pytorch/dist/*.whl ./
COPY --from=build /src/pytorch/xla/dist/*.whl ./
COPY --from=build /dist/torchvision*.whl ./

RUN echo "Installing the following wheels" && ls *.whl
RUN pip install *.whl

# Install the dependencies including libtpu, that's why this needs to happen after wheels are installed.
WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "install_deps"

WORKDIR /

RUN rm -rf /ansible /tmp/wheels
COPY --from=build /dist/*.whl /dist/
