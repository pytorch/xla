ARG python_version=3.8
ARG debian_version=buster
ARG ansible_vars

FROM python:${python_version}-${debian_version} AS build

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

RUN ansible-playbook -vvv playbook.yaml -e "stage=build" -e "${ansible_vars}"

FROM python:${python_version}-${debian_version} AS release

WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

RUN ansible-playbook -vvv playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "install_deps"

WORKDIR /dist
COPY --from=build /dist/*.whl /dist/

RUN echo "Installing the following wheels" && ls /dist/*.whl
RUN pip install *.whl

WORKDIR /
RUN rm -rf /ansible
