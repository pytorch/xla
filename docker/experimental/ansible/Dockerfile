ARG python_version=3.8
ARG debian_version=buster
ARG ansible_vars

FROM python:${python_version}-${debian_version} AS build

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}"

FROM python:${python_version}-${debian_version} AS release

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "install_deps"

WORKDIR /tmp/wheels
COPY --from=build /src/pytorch/dist/*.whl ./
COPY --from=build /src/pytorch/xla/dist/*.whl ./

RUN echo "Installing the following wheels" && ls *.whl
RUN pip install *.whl

WORKDIR /

RUN rm -rf /ansible /tmp/wheels
COPY --from=build /dist/*.whl /dist/

