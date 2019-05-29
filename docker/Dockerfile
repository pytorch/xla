FROM debian:stretch

ARG python_version="3.6"
ARG cloud_build="false"

RUN apt-get update
RUN apt-get install -y git sudo python-pip python3-pip
RUN git clone --recursive https://github.com/pytorch/pytorch

# To get around issue of Cloud Build with recursive submodule update
# clone recursively from pytorch/xla if building docker image with
# cloud build. Otherwise, just use local.
# https://github.com/GoogleCloudPlatform/cloud-builders/issues/435
COPY . /pytorch/xla
RUN if [ "${cloud_build}" = true ]; then cd /pytorch && rm -rf xla && \
	git clone --recursive https://github.com/pytorch/xla; fi

RUN cd pytorch && bash xla/scripts/build_torch_wheels.sh ${python_version}
