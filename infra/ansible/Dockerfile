ARG python_version=3.8
ARG debian_version=bullseye

FROM python:${python_version}-${debian_version} AS build

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}"
RUN ansible-playbook -vvv playbook.yaml -e "stage=build_plugin" -e "${ansible_vars}" --skip-tags=fetch_srcs,install_deps

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
