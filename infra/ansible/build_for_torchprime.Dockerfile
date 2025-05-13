FROM python:${python_version}-${debian_version} AS release

WORKDIR /tmp/wheels
COPY ./*.whl ./

RUN echo "Installing the following wheels" && ls *.whl
RUN pip install *.whl

# Install the dependencies including libtpu.
WORKDIR /ansible
RUN pip install ansible
COPY . /ansible

ARG ansible_vars
RUN ansible-playbook -vvv playbook.yaml -e "stage=release" -e "${ansible_vars}" --tags "install_deps"

WORKDIR /

RUN rm -rf /ansible /tmp/wheels
